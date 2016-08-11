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

    args = parser.parse_args()
    return(args)

def removeNonNumeric(value):
    if not isinstance(value,(int, float)):
        return np.nan
    else:
        return value
def removeCustom(value,exclude):
    for char in exclude:
        if value == char:
            return np.nan
    return value
def removeZero(value):
    if value == 0:
        return np.nan
    else:
        return value
def removeNegative(value):
    if value < 0:
        return np.nan
    else:
        return value
def main(args):    
    logger.info("Configuring R")
    rpy2.robjects.numpy2ri.activate()
    base = importr('base')
    utils = importr('utils')
    robjects.r('library(impute)')

    #bring in data
    dat  = wideToDesign(args.input, args.design, uniqID=args.uniqID, group=args.group)

    k = int(args.k)
    rowCutoff = float(args.rowCutoff)
    colCutoff = float(args.colCutoff)

    #Preprocessing
    logger.info("Preprocessing")
    #Try to convert numbers to float
    logger.info("converting values to floats")
    dat.wide.applymap(float)
    #Remove anything that didnt convert
    logger.info("removing nonnumeric values")
    dat.wide = dat.wide.applymap(removeNonNumeric)
    if args.noZero:
        dat.wide = dat.wide.applymap(removeZero)
    if args.noNegative:
        dat.wide = dat.wide.applymap(removeNegative)
    if args.exclude:
        exclude = args.exclude.split(",")
        dat.wide = dat.wide.applymap(removeCustom,args=exclude)
    if args.strategy == "KNN":
    #Run the KNN Imputation
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
            imputedObject = imputeKNN(data=matrixInR,k=k,rowmax=rowCutoff,colmax=colCutoff)
            sys.stdout = out

            imputedDataAsNumpy = np.array(imputedObject[0])
            imputedDataAsPandas = pandas.DataFrame(imputedDataAsNumpy,index=dat.wide[group.index].index,
                columns=dat.wide[group.index].columns)       
            fixedFullDataset.append(imputedDataAsPandas)

            k = int(args.k) #reset k back to normal if it was modified @127
        pdFull = pandas.concat(fixedFullDataset,axis=1)
    else:
        wideData = dat.wide.as_matrix()
        Imputer(copy=False,axis = 1,strategy=args.strategy).fit_transform(wideData)
        pdFull = pandas.DataFrame(wideData,columns=dat.wide.columns,index=dat.wide.index)
        
    logger.info("creating output")
    pdFull.applymap(float)
    pdFull = pdFull.round(4)
    pdFull.to_csv(args.output, sep="\t")

if __name__ == '__main__':
    # Command line options
    args = getOptions()

    # Activate Logger
    logger = logging.getLogger()

    sl.setLogger(logger)

    main(args)

