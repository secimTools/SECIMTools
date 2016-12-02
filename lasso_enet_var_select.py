################################################################################
# DATE: 2016/November/18
# 
# MODULE: lasso.py
#
# VERSION: 1.0
# 
# AUTHOR:  Miguel Ibarra (miguelib@ufl.edu), Matt Thoburn (mthoburn@ufl.edu).
#
# DESCRIPTION: This runs an Elastic Net or Lasso Test on wide data
#
################################################################################
#Built in packages
import os
import sys
import logging
import argparse
import itertools as it
from argparse import RawDescriptionHelpFormatter

#R
import rpy2
import rpy2.robjects.numpy2ri
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage as STAP

#Local packages
import logger as sl
from interface import wideToDesign

# External Packages
import pandas
import numpy as np
from numpy import genfromtxt



def getOptions(myOpts=None):
    description="""  
    This attempts to impute missing values with K Nearest Neighbors Algorithm
    """
    parser = argparse.ArgumentParser(description=description, 
                                    formatter_class=RawDescriptionHelpFormatter)
    standard = parser.add_argument_group(title='Standard input', 
                                description='Standard input for SECIM tools.')
    standard.add_argument( "-i","--input", dest="input", action='store', 
                        required=True, help="Input dataset in wide format.")
    standard.add_argument("-d" ,"--design",dest="design", action='store', 
                        required=True, help="Design file.")
    standard.add_argument("-id", "--ID",dest="uniqID", action='store', 
                        required=True, help="Name of the column with unique"\
                        " identifiers.")
    standard.add_argument("-g", "--group", dest="group", action='store', 
                        required=False, default=False, help="Name of the column"\
                        " with groups.")
    standard.add_argument("-a", "--alpha", dest="alpha", action="store",
                        required=True, help="Alpha Value.")

    output = parser.add_argument_group(title='Required output')
    output.add_argument("-c", "--coefficients", dest="coefficients", 
                        action="store", required=False, help="Path of en"\
                        " coefficients file.")
    output.add_argument("-f", "--flags", dest="flags", action="store", 
                        required=False, help="Path of en flag file.")
    output.add_argument("-p", "--plots", dest="plots", action="store", 
                        required=False, help="Path of en coefficients file.")                            
    args = parser.parse_args()
    return(args)

def dropMissing(wide):
    """
    Drops missing data out of the wide file

    :Arguments:
        :type wide: pandas.core.frame.DataFrame
        :param wide: DataFrame with the wide file data

    :Returns:
        :rtype wide: pandas.core.frame.DataFrame
        :return wide: DataFrame with the wide file data without missing data
    """
    #Warning
    logger.warn("Missing values were found")

    #Count of original
    nRows = len(wide.index)      

    #Dropping 
    wide.dropna(inplace=True)    

    #Count of dropped
    nRowsNoMiss = len(wide.index)  

    #Warning
    logger.warn("{} rows were dropped because of missing values.".
                format(nRows - nRowsNoMiss))
    return wide

def main(args):
    #Get R ready
    # Get current pathway
    myPath = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    
    # Stablish path for LASSO script
    my_r_script_path = os.path.join(myPath, "lasso_enet.R")
    logger.info(my_r_script_path)

    # Activate pandas2ri
    pandas2ri.activate()

    # Running LASSO R sctrip
    with open(my_r_script_path, 'r') as f:
        rFile = f.read()
    lassoEnetScript = STAP(rFile, "lasso_enet")
    
    # Importing data trought interface
    dat = wideToDesign(args.input, args.design, args.uniqID, group=args.group)

    # Dropping missing values
    # Treat everything as numeric
    dat.wide = dat.wide.applymap(float)

    #Dropping missing values
    if np.isnan(dat.wide.values).any():
        dat.wide = dropMissing(dat.wide)

    # Transpossing data
    dat.trans = dat.transpose()
    dat.trans.columns.name=""

    # Dropping nan columns from design
    removed = dat.design[dat.design[dat.group]== "nan"]
    dat.design = dat.design[dat.design[dat.group] != "nan"]
    dat.trans.drop(removed.index.values, axis=0, inplace=True)

    logger.info("{0} removed from analysis".format(removed.index.values))
    dat.design.rename(columns={dat.group:"group"}, inplace=True)
    dat.trans.rename(columns={dat.group:"group"}, inplace=True)

    #Generate a group List
    groupList = [title for title, group in dat.design.groupby("group") 
                if len(group.index) > 2]

    #Turn group list into pairwise combinations
    comboMatrix = np.array(list(it.combinations(groupList,2)))
    comboLength = len(comboMatrix)

    #Run R
    returns = lassoEnetScript.lassoEN(dat.trans, dat.design, comboMatrix, 
                                    comboLength,args.alpha,args.plots)
    robjects.r['write.table'](returns[0],file=args.coefficients,sep='\t',
                            quote=False, row_names = False, col_names = True)
    robjects.r['write.table'](returns[1],file=args.flags,sep='\t',
                            quote=False, row_names = False, col_names = True)


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    # Activate Logger
    logger = logging.getLogger()
    sl.setLogger(logger)

    # Import data
    logger.info(u"Importing data with the folowing parameters: "\
        "\n\tWide:  {0}"\
        "\n\tDesign:{1}"\
        "\n\tUniqID:{2}"\
        "\n\tAlpha: {3}".\
        format(args.input,args.design,args.uniqID,args.alpha))

    # Run main script
    main(args)
