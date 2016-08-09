#!/usr/bin/env python
# Built-in packages
import re
import copy
import logging
import argparse
from itertools import combinations
from collections import defaultdict
from itertools import cycle, islice
from argparse import RawDescriptionHelpFormatter

# Add-on packages
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.formula.api import ols
from matplotlib.backends.backend_pdf import PdfPages

# Local Packages
import logger as sl
from interface import wideToDesign

#graphing packages
import module_box as box
import module_hist as hist
import module_lines as lines
import module_scatter as scatter
from manager_color import colorHandler
from manager_figure import figureHandler

def getOptions():
    """ Function to pull in arguments """
    description = """ One-Way ANOVA """
    parser = argparse.ArgumentParser(description=description,
                             formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('-i',"--input", dest="input", action='store', 
            required=True, help="Input dataset in wide format.")
    parser.add_argument('-d',"--design", dest="design", action='store', 
            required=True, help="Design file.")
    parser.add_argument('-id',"--ID", dest="uniqID", action='store', 
            required=True, help="Name of the column with unique identifiers.")
    #parser.add_argument('-f',"--formula", dest="formula", action='store', 
    #        required=True, help="Formula to run ANOVA")
    parser.add_argument('-f',"--factors", dest="factors", action='store', 
            required=True, help="Factors to run ANOVA")
    parser.add_argument('-t',"--factorTypes", dest="ftypes", action='store', 
            required=True, help="Type of factors to run ANOVA")
    parser.add_argument('-in',"--interactions", dest="interactions", action="store_true", 
            required=False, help="Ask for interactions to run ANOVA")
    parser.add_argument('-o',"--out", dest="oname", action="store", 
            required=True, help="Output file name.")
    parser.add_argument('-f1',"--fig", dest="ofig", action="store", 
            required=True, help="Output figure name for q-q plots [pdf].")
    parser.add_argument('-f2',"--fig2", dest="ofig2", action="store", 
            required=True, help="Output figure name for volcano plots [pdf].")
    args = parser.parse_args()

    return(args)

def preProcesing(factors,factorType,design):
    factors = factors.split(",")
    factorType = factorType.split(",")
    designCols = design.columns.tolist()
    
    #Check len of the factors
    if len(factors) != len(factorType):
        print "Length of Factors doesnt match FactorType"
        
    #iterating over factor and factorType
    preFormula  = list()
    categorical = list()
    numerical   = list()
    
    # Identify wheter a factor is categorical or numerical
    for factor,fType in zip(factors,factorType):
        if factor not in designCols:
            print "'{}' is not located in your design file".format(factor)
            
        if fType =="C" or fType=="c":
            preFormula.append("C({0})".format(factor))
            categorical.append(factor)
        elif fType =="N" or fType=="n":
            numerical.append(factor)
        else:
            print "'{0}' is not a Valid Flag, use a valid flag to specify \
                    Categorical(C|c) or Numerical (N|n).".format(fType)
    
    # Adding numerical at the end
    preFormula+= numerical
    
    # Creating preFormula
    preFormula =  "+".join(preFormula)
    
    # Get list of unique levels
    levels=[sorted(list(set(design[category].tolist()))) for category \
            in sorted(categorical)]
    
    # Returning 
    return preFormula,categorical,numerical,levels

def parseFormula(f,uid):
    # Remove blank 
    f = re.sub(" ","",f)
    
    # Remove not allowed characters
    f = re.sub("[!|@|#|$|%|^|&||-|_|=|[|]|{|}|\|/|||.|,|;]","",f)
    
    # Convert lowercase c to uppercase C
    f = re.sub("c\(","C(",f)
        

    # Keeping jsut the second part of the formula
    if "~"in f:
        preForm = f.split("~")[1]
    else:
        print "'~' is missing from your formula"
    
    # Replace all alowed  characther with \t to further tokenize
    f2 = re.sub("[~|*|+|:]","\t",f)
    f2 = re.sub("[C(|)]","",f2)
    
    # Getting Uniq tokens
    tokens = list(sorted(set(f2.split("\t"))))
    
    # Remove uid from tokens if fails raise error
    try:
        tokens.remove(uid)
    except:
        print "{0} is not located in your formula, writte your \
                formula again and make sure your first element match \
                your unique ID.".format(uid)    
        
    # Getting indexes for given token    
    tokDict = {tok:"{"+str(i)+"}"for i,tok in enumerate(tokens)}
    
    # Substitute values based on tokDict
    form = re.sub("|".join(tokens),lambda x: tokDict[x.group(0)],preForm)
    
    # Return parsed formula scheme without the metabolite and group 
    #of uniq tokens
    return "~"+form.format(*tokens), tokens

def startResults(wide,design,groups):
    # Opening results dataframe
    results = pd.DataFrame(index=wide.index)

    # Get grandMean
    results["GrandMean"] = wide.mean(axis=1)

    # Get variance
    results["SampleVariance"]= wide.var(axis=1)

    # Get group means 
    for group in groups:
        for lvlName,subset in design.groupby(group):
            results["Mean {0}-{1}".format(group,lvlName)] = \
            wide[subset.index].mean(axis=1)
            
    # Return results
    return results

def generateDinamicCmbs(factors,globList,acum=False):
    # If not factors left return None
    factor = factors.pop()
    
    # If factor found iterate over it
    for level in factor:
        if len(factors)>0:
            if acum:
                if acum[-1] in factor:
                    acum[-1] = level
                else:
                    acum=acum+[level]
            else:
                acum = [level]
            
            # If any factor left then recurse again
            generateDinamicCmbs(copy.deepcopy(factors),globList,acum)
            
        else:
            if acum:
                if acum[-1] in factor:
                    finalList[-1] = level
                else:
                    finalList=acum+[level]
            else:
                finalList = [level]
            
            globList.append(finalList)

def fuckingANOVA(data, fx, combs, metName, factors, cutoff=4):

    # Creating list for fullRes and IndexToDrop
    fullRes     =list()
    indexToDrop =list()

    # Make a copy of the levels on the factors(groups)
    groups=copy.copy(combs)

    # Reverse list
    groups.reverse()

    # Take one element of the group and pop it
    while len(groups)>0:
        # Take las element of the list 
        elem =  groups.pop()
        
        # Create tempDF to change Order
        tempDF = changeOrder(data,elem,factors)
        
        # Running ANOVA on data
        anova = ols(formula=fx, data=tempDF).fit()

        # Saving a dataframe for anova results
        aGrpRes = anovaResultsByGroup(anova,"|".join(elem))
        
        # Dropping duplicates
        aGrpRes = removeAnovaDups(indexToDrop,df=aGrpRes)

        # Appending results to list
        fullRes.append(aGrpRes)

        # Appending current indexes to indextoDrop list
        indexToDrop= indexToDrop+aGrpRes.index.tolist()
        
    # Creating one df with all the results
    fullRes = pd.concat(fullRes)

    # Calculating -log10 of pval
    fullRes["log10(p-value)"] = -np.log10(fullRes["Prob>|t| for Diff"])

    # Flagging lpval > cuttof
    fullRes["Sig Index for Diff"] = np.where(np.abs(fullRes["log10(p-value)"])\
                                             > cutoff,int(0),int(1));fullRes

    # Stacking results
    fullRes = fullRes.T.stack(level=0)

    # Getting general results for anova
    aRes = anovaResults(anova)

    # We need to create a Series with all the results values
    index=[]
    data=[]
    # For general data
    for result in aRes.index.tolist():
        index.append(result)
        data.append(aRes[result])

    # For group data
    for i,j in fullRes.index:
        index.append("{0} {1}".format(i,j))
        data.append(fullRes[i][j])

    # Creating dataframe 
    fullRes = pd.Series(data=data, index=index, name=metName)

    # Return pd.Series with full results 
    return fullRes

def changeOrder(data,combN,factors):
    # Makje a copy of trans
    tempDF = copy.deepcopy(data)

    # Generate new names for groupst this way is possible to change the order 
    #  of  "intercept" in anova
    for elem in combN:
        # NewGrpNames for current factor
        newGrpNames = ["1_"+lvl if lvl==elem else lvl for lvl in tempDF[factors[combN.index(elem)]]]

        # Replace old nams with new names
        tempDF[factors[combN.index(elem)]]=newGrpNames
    
    # Returning results
    return tempDF

def anovaResultsByGroup(model, elem):
    # Extracting the parameters we are interested in from ANOVA
    # These values are going to be used multiple times
    coef = -(model.params)
    stde = model.bse
    t    = model.tvalues
    pt   = model.pvalues
        
    #Add name to previous series
    t.name    ="t-Value for Diff"
    stde.name ="StdError for Diff"
    coef.name ="Diff of"
    pt.name   ="Prob>|t| for Diff"

    # Concat all dataframes
    df = pd.concat([coef,stde,t,pt],axis=1)
    
    # Removing intercepts
    df.drop("Intercept",inplace=True,axis="index")
    
    # Creating pretty names for indexes
    oldIndex = dict()
    for origIndx in df.index:
        if origIndx in ["runOrder"]:
            df.drop(origIndx,inplace=True)        
        else:
            modIndx = re.sub(".+\[T\.|\]","",origIndx)
            oldIndex[origIndx] = "{0}-{1}".format(elem,modIndx)
            print origIndx
    
    #Rename indexs
    df.rename(index=oldIndex, inplace=True)

    #Returns
    return df

def anovaResults(model):
    # Getting results
    f      = model.fvalue
    p      = model.f_pvalue
    mss    = model.ssr
    ess    = model.ess
    tss    = mss+ess
    mse    = model.mse_resid
    NDF    = int(model.df_model)
    DDF    = int(model.df_resid)
    R2     = model.rsquared
    resid  = model.resid/np.sqrt(mse)
    fitted = model.fittedvalues
    
    # Puttign al the results together
    anovaResults=[f,p,mss,ess,tss,mse,NDF,DDF,R2,resid,fitted]
    
    # Creating indexes for values
    index=["f-Value","p-Value of f-Value",
        "ModelSS","ErrorSS","TotalSS",
        "MSE","NDF","DDF", "R2","resid","fitted"]
    
    # Creating results series
    results = pd.Series(data=anovaResults, index=index)
    
    # Return series
    return results

def removeAnovaDups(toDrop,df):
    # For every index in the df list
    for idx in df.index.tolist():
        
        # Split index using "-"
        idxElements =  idx.split("-")
        
        # For element to drop in toDrop
        for drop in toDrop:
            
            # Split element to drop by "-"
            d1,d2 = drop.split("-")
            
            # If element to drop is in df index
            if (d1 in idxElements) and (d2 in idxElements):
                
                # Drop df row on that index
                df = df.drop(idx,axis="index")
                
    # Return df
    return df

def qqPlot(tresid, tfit, oname):
    """ 
    Plot the residual diagnostic plots by sample.

    Output q-q plot, boxplots and distributions of the residuals. These plots
    will be used diagnose if residuals are approximately normal.

    :Arguments:
        :type tresid: pandas.Series
        :param tresid: Pearson normalized residuals. (transposed)
                        (residuals / sqrt(MSE))

        :type tfit: pandas DataFrame
        :param tfit: output of the ANOVA (transposed)

        :type oname: string
        :param oname: Name of the output file in pdf format.

    :Returns:
        :rtype: PDF
        :returns: Outputs a pdf file containing all plots.

    """
    #Open pdf
    with PdfPages(oname) as pdf:

        # Stablishing axisLayout
        axisLayout = [(0,0,1,1),(0,1,1,1),(0,2,1,1),(1,0,3,1)]

        # Start plotting
        for col in tresid.columns:
            #Creating figure
            fig = figureHandler(proj='2d',numAx=4,numRow=2,numCol=3,
                                arrangement=axisLayout)


            data = tresid[col].values.ravel()
            noColors = list()
            for j in range(0,len(data)):
                noColors.append('b')#blue
            df_data = pd.DataFrame(data)

            # Plot qqplot on axis 0
            sm.graphics.qqplot(tresid[col],fit=True,line='r',ax=fig.ax[0])

            # Plot boxplot on axis 1
            box.boxSeries(ser=data,ax=fig.ax[1])

            # Plot histogram on axis 2
            hist.quickHist(ax=fig.ax[2],dat=df_data,orientation='horizontal')

            # Plot scatterplot on axis 3
            scatter.scatter2D(ax=fig.ax[3],x=tfit[col], y=tresid[col],
                                colorList=list('b'))

            # Draw cutoff line for scatterplot on axis 3
            lines.drawCutoffHoriz(ax=fig.ax[3],y=0)

            # Format axis 0
            fig.formatAxis(figTitle=col,axnum=0,grid=False,showX=False,
                yTitle="Sample Quantiles")

            # Format axis 1
            fig.formatAxis(axnum=1,axTitle="Distribution of P-values",
                grid=False,showX=False,showY=False)

            # Format axis 2
            fig.formatAxis(axnum=2,showX=False,showY=False)

            # Format axis 3
            fig.formatAxis(axnum=3,axTitle="Fitted Values vs Residuals",
                xTitle="Fitted Values",yTitle="Standardized Residuals",
                grid=False)

            #Add figure to pdf
            fig.addToPdf(pdfPages=pdf)

def volcano(combo, results, oname, cutoff=4):
    """ 
    Plot volcano plots.

    Creates volcano plots to compare means, for all pairwise differences.

    :Arguments:

        :type combo: dictionary
        :param combo: A dictionary of dictionaries with all possible pairwise
            combinations. Used this to create the various column headers in the
            results table.

        :type results: pandas DataFrame
        :param results: TODO

        :type oname: string
        :param oname: Name of the output file in pdf format.
       
        :type cutoff: int
        :param cutoff: The cutoff value for significance.

    :Returns:
        :rtype: PD
        :returns: Outputs a pdf file containing all plots.

    """
    # Getting data for lpvals
    lpvals = {col.split(" ")[-1]:results[col] for col in results.columns.tolist() \
            if col.startswith("log10(p-value)")}

    # Gettign data for diffs
    difs   = {col.split(" ")[-1]:results[col] for col in results.columns.tolist() \
            if col.startswith("Diff of")}
    

    # Making plots
    with PdfPages(oname) as pdf:
        for key in sorted(difs.keys()):
            # Set Up Figure
            volcanoPlot = figureHandler(proj="2d")

            # Plot all results
            scatter.scatter2D(x=list(difs[key]), y=list(lpvals[key]), 
                                colorList=list('b'), ax=volcanoPlot.ax[0])

            # Color results beyond treshold red
            cutLpvals = lpvals[key][lpvals[key]>cutoff]
            if not cutLpvals.empty:
                cutDiff = difs[key][cutLpvals.index]
                scatter.scatter2D(x=list(cutDiff), y=list(cutLpvals), 
                                colorList=list('r'), ax=volcanoPlot.ax[0])

            # Drawing cutoffs
            lines.drawCutoffHoriz(y=cutoff, ax=volcanoPlot.ax[0])

            # Format axis (volcanoPlot)
            volcanoPlot.formatAxis(axTitle=key,grid=False)

            # Add figure to PDF
            volcanoPlot.addToPdf(pdfPages=pdf)

def main(args):
    # Parse Formula
    # preForm,factors = parseFormula(f=args.formula,uid=args.uniqID)

    # Import data
    dat = wideToDesign(args.input,args.design,args.uniqID)

    # Generate formula Formula
    preFormula,categorical,numerical,levels = preProcesing(factors=args.factors,
                        factorType=args.ftypes,design=dat.design)

    # Transpose data
    dat.trans  = dat.transpose()

    # Create combination of groups
    levels.reverse()
    lvlComb = list()
    generateDinamicCmbs(levels,lvlComb)

    # Maps every metabolite to its formulas
    dictFormula = {feature:"{0}~{1}".format(str(feature),preFormula) for feature \
                    in dat.wide.index.tolist()}

    # Getting grandMean, variance and mean per groups
    results = startResults(wide=dat.wide,design=dat.design,groups=categorical)

    # Creating a list of anova results for each metabolite
    modelResults=list()

    #for feature in dat.wide.index.tolist():
    for feat in dictFormula.keys():
        
        #Run fucking ANOVA
        modelResults.append(fuckingANOVA(data=dat.trans,fx=dictFormula[feat], 
                            combs=lvlComb,metName=feat,factors=categorical))

    # Concatenating results lists into a dataframe
    modelResults=pd.concat(modelResults, axis=1)

    # Transpose modelResults dataframe
    modelResults = modelResults.T

    # Creating pd.Series for Ressids and Fitted Values
    residDat = pd.concat(modelResults["resid"].tolist(), axis=1)
    fitDat = pd.concat(modelResults["fitted"].tolist(), axis=1)


    # print modelResults.columns.tolist()
    # Removing Ressids and Fitted values from modelResults
    modelResults.drop(["fitted","resid"],axis=1,inplace=True)

    # Transpose modelResults and concatenate with results 
    results = pd.concat([results,modelResults], axis=1)

    # QQ plots    
    logger.info('Generating q-q plots.')
    qqPlot(residDat.T, fitDat.T, args.ofig)

    # Generate Volcano plots
    logger.info('Generating volcano plots.')
    volcano(lvlComb, results, args.ofig2)

    # If interactions
    if args.interactions:
        # Merging all categories into one for interactions
        dat.trans["_treatment_"] = dat.trans.apply(lambda x: \
                                "_".join(map(str,x[categorical].values)),axis=1)

        # if numerical adde then to the formula
        if len(numerical)>0:
            formula = ["C(_treatment_)"]+numerical
        else:
            formula = ["C(_treatment_)"]

        # Concatenatig the formula
        formula = "+".join(formula)

        # Getting new formula for interactions
        dictFormula = {feature:"{0}~{1}".format(str(feature),formula) \
                    for feature in dat.wide.index.tolist()}

        # Creating list for interactions
        interactions= list()

        # Creating levelCombinations
        levels=sorted(list(set(dat.trans["_treatment_"].tolist())))

        # Creating levelCombinations
        lvlComb = list()
        generateDinamicCmbs([levels],lvlComb)

        #for feature in dat.wide.index.tolist():
        for feat in dictFormula.keys():
        
            #Run fucking ANOVA
            interactions.append(fuckingANOVA(data=dat.trans,fx=dictFormula[feat],
                                combs=lvlComb,metName=feat,factors=["_treatment_"]))

        # Concatenating interaction results lists into a dataframe
        interactions=pd.concat(interactions, axis=1)

        # Transpose dataFrame
        interactions = interactions.T

        # Dropping fitted and resid
        interactions.drop(["fitted","resid"],axis=1,inplace=True)
       
        # Transpose modelResults and concatenate with results 
        results = pd.concat([results,interactions], axis=1)

    # Round results to 4 digits and save
    results = results.round(4)
    results.to_csv(args.oname, sep="\t")

if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    sl.setLogger(logger)

    logger.info(u"""Importing data with following parameters: \
        \n\tWide: {0}\
        \n\tDesign: {1}\
        \n\tUnique ID: {2}\
        \n\tFactors: {3}"""
        .format(args.input, args.design, args.uniqID, args.factors))

    main(args)