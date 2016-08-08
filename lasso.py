import numpy as np
from numpy import genfromtxt
import pandas

#R
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage as STAP
from rpy2.robjects import pandas2ri
#Add Ons


#Local packages
from interface import wideToDesign
import logger as sl
#Built in packages
import logging
import argparse
from argparse import RawDescriptionHelpFormatter
import sys
import itertools as it

import os

def getOptions(myOpts=None):
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
    standard.add_argument("-a","--alpha",dest="alpha",action="store",required=True,
                        help="TODO")

    output = parser.add_argument_group(title='Required output')
    output.add_argument("-c","--coefficients",dest="coefficients",action="store",required=False,
                        help="Path of en coefficients file.")
    output.add_argument("-f","--flags",dest="flags",action="store",required=False,
                        help="Path of en flag file.")
    output.add_argument("-p","--plots",dest="plots",action="store",required=False,
                        help="Path of en coefficients file.")                            
    args = parser.parse_args()
    return(args)

def main(args):
    #Get R ready
    #os.path.dirname(os.path.realpath("alexScript2.R"))
    os.chdir("/home/mthoburn/Desktop/work/galaxy/tools/GalaxyTools")
    #print os.getcwd()
    pandas2ri.activate()
    with open('alexScript2.R', 'r') as f:
        rFile = f.read()
    alexScript = STAP(rFile, "alexScript2")
    
    #Input
    dat = args.input
    design = args.design
    uniqID = args.uniqID
    group = args.group
    alpha = args.alpha
    
    coef = args.coefficients
    flags = args.flags
    plots = args.plots

    data = wideToDesign(dat,design,uniqID,group=group)
    data.trans = data.transpose()
    data.trans.columns.name=""


    #data.trans.to_csv("test-data/alex.tsv",sep='\t')
    #Generate a group List
    datG = data.design.groupby(data.group)
    groupList = list()
    for title, group in datG:
        if len(group.index) > 2:
            groupList.append(title)
    #print groupList

    #Turn group list into pairwise combinations
    comboMatrix = np.array(list(it.combinations(groupList,2)))
    comboLength = len(comboMatrix)


    #Run R
    returns = alexScript.lassoEN(data.trans,data.design,comboMatrix,comboLength,alpha,plots)
    robjects.r['write.table'](returns[0],file=coef,sep='\t')
    robjects.r['write.table'](returns[1],file=flags,sep='\t')
    #pandas2ri.ri2pandas(returns[1]).to_csv("test-data/en_flags.tsv",sep='\t')


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    # Activate Logger
    logger = logging.getLogger()

    sl.setLogger(logger)

    main(args)