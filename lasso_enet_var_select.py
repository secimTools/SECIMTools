################################################################################
# DATE: 2017/03/10
# 
# MODULE: lass_enet_var_select.py
#
# VERSION: 1.1
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
    # Standard input
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
    # Tool especific
    tool = parser.add_argument_group(title='Tool Especific')
    tool.add_argument("-a", "--alpha", dest="alpha", action="store",
                        required=True, help="Alpha Value.")
    # Output
    output = parser.add_argument_group(title='Required output')
    output.add_argument("-c", "--coefficients", dest="coefficients", 
                        action="store", required=False, help="Path of en"\
                        " coefficients file.")
    output.add_argument("-f", "--flags", dest="flags", action="store", 
                        required=False, help="Path of en flag file.")
    output.add_argument("-p", "--plots", dest="plots", action="store", 
                        required=False, help="Path of en coefficients file.")                            
    args = parser.parse_args()

    # Standardize paths
    args.input        = os.path.abspath(args.input)
    args.plots        = os.path.abspath(args.plots)
    args.flags        = os.path.abspath(args.flags)
    args.design       = os.path.abspath(args.design)
    args.coefficients = os.path.abspath(args.coefficients)

    return(args)

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
    dat = wideToDesign(args.input, args.design, args.uniqID, group=args.group,
                        logger=logger)

    # Cleaning from missing data
    dat.dropMissing()

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
    # Finishing
    logger.info("Script Complete!")

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
