######################################################################################
# DATE: 2016/August/10
# 
# MODULE: imputation.py
#
# VERSION: 1.0
# 
# AUTHOR: Matt Thoburn (mthoburn@ufl.edu) ed. Miguel Ibarra (miguelib@ufl.edu)
#
# DESCRIPTION: This attempts to impute missing data by an algorithm of the user's choice
#
#######################################################################################
#R
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
import rpy2.robjects.numpy2ri

#Add Ons
import numpy as np
from numpy import genfromtxt
import pandas

#Bayesian PYMC
from pymc.distributions import Impute
from pymc import Poisson, Normal, DiscreteUniform
import pymc
from pymc import MCMC
from pymc.distributions import Impute


#Local packages
from interface import wideToDesign
import logger as sl
#Built in packages
import logging
import argparse
from argparse import RawDescriptionHelpFormatter
import sys
from sklearn.preprocessing import Imputer 

def getOptions(myOpts = None):
    description="""  
    This attempts to impute missing values with K Nearest Neighbors Algorithm
    """
    parser = argparse.ArgumentParser(description=description, 
                                    formatter_class=RawDescriptionHelpFormatter)
    standard = parser.add_argument_group(title='Standard input', 
                                description='Standard input for SECIM tools.')
    standard.add_argument( "-i","--input", dest="input", action='store', required=True, 
                        help="Input dataset in wide format.")
    standard.add_argument("-d" ,"--design",dest="design", action='store', required=True,
                        help="Design file.")
    standard.add_argument("-id", "--ID",dest="uniqID", action='store', required=True, 
                        help="Name of the column with unique identifiers.")
    standard.add_argument("-g", "--group",dest="group", action='store', required=False, 
                        default=False,help="Name of the column with groups.")
    standard.add_argument("-s","--strategy",dest="strategy",action="store",required=True,
                        help="Imputation strategy: KNN, mean, median, or most frequent")

    output = parser.add_argument_group(title='Required output')
    output.add_argument("-o","--output",dest="output",action="store",required=False,
                        help="Path of output file.")

    optional = parser.add_argument_group(title='Optional input')
    optional.add_argument("-noz","--no_zero",dest="noZero",action='store_true',
                        required=False,default=True,help="Treat 0 as missing?")
    optional.add_argument("-noneg","--no_negative",dest="noNegative",action='store_true',
                        required=False,default=True,help="Treat negative as missing?")
    optional.add_argument("-ex","--exclude",dest="exclude",action='store',
                        required=False,default=False,help="Additional values to treat as missing data, seperated by commas")

    optional.add_argument("-k","--knn",dest="k",action='store',
                        required=False,default=5,help="Number of nearest neighbors to search Default: 5.")
    optional.add_argument("-rc","--row_cutoff",dest="rowCutoff",action='store',
                        required=False,default=.5,help="Percent cutoff for imputation of rows."
                        "If this is exceeded, imputation will be done by mean instead of knn. Default: .5")
    optional.add_argument("-cc","--col_cutoff",dest="colCutoff",action='store',
                        required=False,default=.8,help="Percent cutoff for imputation of columns. "
                        "If this is exceeded, imputation will be done by mean instead of knn. Default: .8")
    optional.add_argument("-bc","--bayes_cutoff",dest="bayesCutoff",action='store',
                        required=False,default=.5,help="If you are not using the Bayesian Strategy, " \
                        "then ignore this. For a given row, if this ratio of missing to present values is exceeded, " \
                        "imputation will be skipped. Must be greater than 0 and less than 1")
    optional.add_argument("-mu","--mu_method",dest="muMethod",required=False,default="mean",help="use mean or median to " \
                        "generate mu value for bayesian imputation")
    optional.add_argument("-dist","--distribution",dest="dist",required=False,default="Poisson",help="use mean or median to " \
                        "generate mu value for bayesian imputation")
    args = parser.parse_args()
    return(args)
def removeNonNumeric(value):
    """If value cannot be read as a number, replace with not a number"""
    if not isinstance(value,(int, float)):
        return np.nan
    else:
        return value
def removeCustom(value,exclude):
    """If value is a custom character specified by user, replace with not a number"""
    for char in exclude:
        if value == char:
            return np.nan
    return value
def removeZero(value):
    """If value is 0, replace with not a number"""
    if value == 0:
        return np.nan
    else:
        return value
def removeNegative(value):
    """If value is negative, replace with not a number"""
    if value < 0:
        return np.nan
    else:
        return value
def preprocess(noz,non,ex,data):
    """
    Preprocesses data to replace all unaccepted values with np.nan so they can be imputedDataAsNumpy

    :Arguments:
        :type noz: bool
        :param noz: remove zeros?

        :type non: bool
        :param non: remove negative numbers?

        :type ex: string
        :param ex: custom characters to be removed. Will be parsed to a list of strings

        :type data: pandas DataFrame
        :param data: data to be imputed

    :Return:
        :type data: pandas DataFrame
        :param data: data to be imputed
    """
    data = data.applymap(float)
    data = data.applymap(removeNonNumeric)
    if noz:
        data = data.applymap(removeZero)
    if non:
        data = data.applymap(removeNegative)
    if ex:
        exclude = ex.split(",")
        data = data.applymap(removeCustom,args=exclude)
    return data
def imputeKNN(rc,cc,k,dat):
    """
    Imputes by K-Nearest Neighbors algorithm

    :Arguments:
        :type rc: float
        :param rc: row cutoff value that determines whether or not to default to mean imputation

        :type cc: float
        :param cc: column cutoff value that determines wheter or not to default to mean imputation

        :type k: int
        :param k: Number of nearby neighbors to consider when performing imputation

        :type dat: interface wideToDesign file
        :param dat: wide and design data bundled together

    :Returns:
        :type pdFull: pandas DataFrame
        :param pdFull: data with missing values imputed
    """
    logger.info("Configuring R")
    rpy2.robjects.numpy2ri.activate()
    base = importr('base')
    utils = importr('utils')
    robjects.r('library(impute)')

    datG = dat.design.groupby(dat.group)
    fixedFullDataset = list()

    out = sys.stdout #Save the stdout path for later, we're going to need it
    f = open('/dev/null','w') #were going to use this to redirect stdout temporarily
    logger.info("running imputation")
    for title, group in datG:

        groupLen = len(group.index)
        if groupLen == 1: #No nearby neighbors to impute
            logger.info(title + " has no neighbors, will not impute")
            fixedFullDataset.append(dat.wide[group.index])
            continue
        elif groupLen <= k: #some nearby, but not enough to use user specified k
            logger.info(title + " group length less than k, will use group length - 1 instead")
            k = groupLen - 1
            

        wideData = dat.wide[group.index].as_matrix()
        numRows, numCols = wideData.shape

        matrixInR = robjects.r['matrix'](wideData,nrow=numRows,ncol=numCols)
        imputeKNN = robjects.r('impute.knn')
        sys.stdout = f
        imputedObject = imputeKNN(data=matrixInR,k=k,rowmax=rc,colmax=cc)
        sys.stdout = out

        imputedDataAsNumpy = np.array(imputedObject[0])
        imputedDataAsPandas = pandas.DataFrame(imputedDataAsNumpy,index=dat.wide[group.index].index,
            columns=dat.wide[group.index].columns)       
        fixedFullDataset.append(imputedDataAsPandas)

        k = int(args.k) #reset k back to normal if it was modified @125
        pdFull = pandas.concat(fixedFullDataset,axis=1)
        return pdFull
def imputeBayesian(bc,dist,mu,dat):
    """
    Imputes by Bayesian Probability algorithm
    
    :Arguments:
        :type bc: float
        :param bc: row cutoff value which determines whether or not to leave data as missing

        :type dist: string
        :param dist: Distribution, normal or Poisson

        :type mu: string
        :param mu: method to determine mu value. Can be mean or median

        :type dat: interface wideToDesign file
        :param dat: wide and design data bundled together

    :Returns:
        :type pdFull: pandas DataFrame
        :param pdFull: data with missing values imputed 
    """
    datG = dat.design.groupby(dat.group)
    out = sys.stdout #Save the stdout path for later, we're going to need it
    f = open('/dev/null','w') #were going to use this to redirect stdout temporarily
    bigDFList = list()
    singleGroupCols = list()
    singleGroupNames = list()
    for title, group in datG:

        groupLen = len(group.index)
        if groupLen == 1: #No nearby neighbors to impute
            singleGroupCols.append(dat.wide[group.index])
            singleGroupNames.append(group.values[0][0])
            continue
        else:
            rowBuilder = list()
            metBuilder = list()
            for index, row in dat.wide[group.index].iterrows():
                if 0 in row.values:
                #if np.any(np.isnan(row.values)):
                    values = row.values

                    missing = list()                
                    for i in range(len(values)):
                        if values[i] == 0:
                            missing.append(i)
                    #If entire row is missing, skip
                    if float(len(missing))/len(values) == 1:
                        #logger.info("All values missing")
                        rowBuilder.append(row)
                        continue
                    #If ratio of missing to expected is greater than cutoff, skip  
                    if float(len(missing))/len(values) > .5:
                        #logger.info("too many missing")
                        rowBuilder.append(row)
                        continue

                    valuesMasked = np.ma.masked_equal(values,value=0)
                    #print valuesMasked
                    sys.stdout = f

                    if dist == "Normal":
                        if np.std(valuesMasked) == 0:
                            tau = np.square(1/(np.mean(valuesMasked)/3))
                        else:    
                            tau = np.square((1/(np.std(valuesMasked))))
                        if mu == "mean":
                            x = Impute('x',Normal,valuesMasked,mu=np.mean(valuesMasked),tau=tau)
                        else:
                            x = Impute('x',Normal,valuesMasked,mu=np.median(valuesMasked),tau=tau)    
                    else:
                            x = Impute('x',Poisson,valuesMasked,mu=np.mean(valuesMasked))

                    m = MCMC(x)
                    m.sample(iter=1,burn=0,thin=1)
                    sys.stdout = out

                    for i in range(len(missing)):
                        keyString = "x[" + str(missing[i]) + "]"
                        imputedValue = m.trace(keyString)[:]
                        row.iloc[missing[i]] = imputedValue[0]
                rowBuilder.append(row)

            smallDF = pandas.concat(rowBuilder,axis=1)
            bigDFList.append(smallDF)

    bigDF = pandas.concat(bigDFList)
    pdFull = bigDF.transpose()

    #we have a list of single groups which need to get added back
    for i in range(len(singleGroupNames)):
        pdFull[singleGroupNames[i]] = singleGroupCols[i]
    return pdFull    
def imputeLazy(dat,strategy):
    bigDFList = list()
    singleGroupCols = list()
    singleGroupNames = list()
    datG = dat.design.groupby(dat.group)

    for title, group in datG:

        groupLen = len(group.index)
        if groupLen == 1: #No nearby neighbors to impute
            singleGroupCols.append(dat.wide[group.index])
            singleGroupNames.append(group.values[0][0])
            continue
        else:
            rowBuilder = list()
            metBuilder = list()
            for index, row in dat.wide[group.index].iterrows():
                if np.any(np.isnan(row.values)):
                    values = row.values
                    missing = list()                
                    for i in range(len(values)):
                        if values[i] == np.nan:
                            missing.append(i)
                    #If entire row is missing, skip
                    if float(len(missing))/len(values) == 1:
                        #logger.info("too many missing")
                        rowBuilder.append(row)
                        continue
                    #If ratio of missing to expected is greater than cutoff, skip  
                    if float(len(missing))/len(values) > .5:
                        #logger.info("too many missing")
                        rowBuilder.append(row)
                        continue

                    mean = np.nanmean(values)
                    median = np.nanmedian(values)
                    #print mean
                    #print values
                    for i in range(len(missing)):
                        if strategy == "Mean":
                            row.iloc[missing[i]] = mean
                        elif strategy == "Median":
                            row.iloc[missing[i]] = median

                rowBuilder.append(row)

            smallDF = pandas.concat(rowBuilder,axis=1)
            bigDFList.append(smallDF)

    bigDF = pandas.concat(bigDFList)
    pdFull = bigDF.transpose()

    #we have a list of single groups which need to get added back
    for i in range(len(singleGroupNames)):
        pdFull[singleGroupNames[i]] = singleGroupCols[i]
    
    return pdFull
def main(args):
    #bring in data
    dat  = wideToDesign(args.input, args.design, uniqID=args.uniqID, group=args.group)

    k = int(args.k)
    rowCutoff = float(args.rowCutoff)
    colCutoff = float(args.colCutoff)
    bayesCutoff = float(args.bayesCutoff)
    muMethod = args.muMethod
    distribution = args.dist
    #Preprocessing
    logger.info("Preprocessing")
    #dat.wide = preprocess(noz=args.noZero,non=args.noNegative,ex=args.exclude,data=dat.wide)
    #dat.wide.to_csv("test-data/preprocess.tsv",sep="\t")
    if args.strategy == "KNN":
        pdFull = imputeKNN(rc=rowCutoff,cc=colCutoff,k=k,dat=dat)
    elif args.strategy == "Bayesian":
        pdFull = imputeBayesian(bc=bayesCutoff,dist=distribution,mu=muMethod,dat=dat)
    else:
        pdFull = imputeLazy(dat=dat,strategy=args.strategy) 
        
    logger.info("creating output")
    pdFull.applymap(float)
    pdFull = pdFull.round(4)
    pdFull.index.name = args.uniqID
    pdFull.to_csv(args.output, sep="\t")

if __name__ == '__main__':
    # Command line options
    args = getOptions()
    # Activate Logger
    logger = logging.getLogger()
    sl.setLogger(logger)
    main(args)

