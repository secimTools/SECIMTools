# Import build-in librearies
import copy

# import add-on packages
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

# Importing anova packages
from anovaModules.preProcessing import preProcessing
from anovaModules.changeDFOrder import changeDFOrder
from anovaModules.getModelResults import getModelResults
from anovaModules.startANOVAResults import startANOVAResults
from anovaModules.generateDinamicCmbs import generateDinamicCmbs
from anovaModules.removeAnovaDupResults import removeAnovaDupResults
from anovaModules.getModelResultsByGroup import getModelResultsByGroup

def runANOVA(dat, formula, lvlComb, categorical, levels, numerical, cutoff=4):
    """
    Core for processing all the data.

    :Arguments:
        :type dat: wideToDesign object.
        :param dat: wide, design, group, anno, trans.

        :type formula: dictionary 
        :param formula: Contains the formulas in a row:formula fashion.

        :type lvlComb: list.
        :param lvlComb: list with all the levels in the factors.

        :type categorical: list.
        :param categorical: Contains the names of the categorical factors.

        :type levels: list.
        :param levels: Name of the .
        
        :type numerical: list.
        :param numerical: Contains the names of the numerical factors.

    :Returns:
        :rtype results: pd.DataFrames
        :return results: dataframe in wide format with the results of the model

        :rtype residDat: pd.DataFrames
        :return residDat: Contains the residuals of the model
        
        :rtype fitDat: pd.DataFrames
        :return fitDat: dataframe with all the fitted data
    """

    # Getting grandMean, variance and mean per groups
    results = startANOVAResults(wide=dat.wide,design=dat.design,groups=categorical)

    # Creating a list of anova results for each metabolite
    modelResults=list()

    # iterating over all the formula keys
    for feat in formula.keys():
        combs=copy.copy(lvlComb)

        # Creating list for fullRes and IndexToDrop
        fullRes     =list()
        indexToDrop =list()

        # Reverse list
        combs.reverse()

        # Take one element of the group and pop it
        while len(combs)>0:
            # Take las element of the list 
            elem =  combs.pop()
            
            # Create tempDF to change Order
            tempDF = changeDFOrder(data=dat.trans, combN=elem, factors=categorical)
            
            # Running ANOVA on data
            anova = ols(formula=formula[feat], data=tempDF).fit_regularized()

            # Saving a dataframe for anova results
            aGrpRes = getModelResultsByGroup(anova,levels,numerical)
            
            # Dropping duplicates
            aGrpRes = removeAnovaDupResults(indexToDrop,df=aGrpRes)

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
        aRes = getModelResults(anova)

        # We need to create a Series with all the results values
        index=[]
        data=[]

        # For general data
        for result in aRes.index.tolist():
            index.append(result)
            data.append(aRes[result])

        # For group data
        for indexName,combName in fullRes.index:
            index.append("{0} {1}".format(indexName,combName))
            data.append(fullRes[indexName][combName])

        # Creating dataframe 
        fullRes = pd.Series(data=data, index=index, name=feat)

        # Append to all results lits
        modelResults.append(fullRes)

    # Concatenating results lists into a dataframe
    modelResults=pd.concat(modelResults, axis=1)

    # Transpose modelResults dataframe
    modelResults = modelResults.T

    # Creating pd.Series for Ressids and Fitted Values
    residDat = pd.concat(modelResults["resid"].tolist(), axis=1)
    fitDat = pd.concat(modelResults["fitted"].tolist(), axis=1)

    # Removing Ressids and Fitted values from modelResults
    modelResults.drop(["fitted","resid"],axis=1,inplace=True)

    # Transpose modelResults and concatenate with results 
    results = pd.concat([results,modelResults], axis=1)

    # Return results
    return results, residDat, fitDat