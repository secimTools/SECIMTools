#!/usr/bin/env python
################################################################################
# DATE: 2017/03/08
# 
# MODULE: multiple_testing_adjustment.py
#
# VERSION: 1.1
# 
# AUTHOR: Miguel Ibarra (miguelib@ufl.edu) 
#
# DESCRIPTION: Adjust pvalues. 
#
################################################################################
# Import built-in libraries
import os
import logging
import argparse

# Import add-on libraries
import numpy as np
import pandas as pd
import statsmodels.sandbox.stats.multicomp as stm

# Import local data libraries
from secimtools.dataManager import logger as sl
from secimtools.dataManager.flags import Flags
from secimtools.dataManager.interface import wideToDesign

def getOptions():
    """Function to pull arguments"""
    parser = argparse.ArgumentParser(description="")
    # Standar Input
    standar = parser.add_argument_group(title='Standard input', 
                        description='Standard input for SECIM tools.')
    standar.add_argument("-i","--input",dest="input", action='store', 
                        required=True, help="Input dataset in wide format.")
    standar.add_argument("-id","--uniqID",dest="uniqID",action="store",
                        required=True, help="Name of the column with unique" \
                        "dentifiers.")
    # Tool Input
    tool = parser.add_argument_group(title='Tool input', 
                        description='Specific input for the tool.')
    tool.add_argument("-pv","--pval",dest="pval",action="store",
                        required=True, help="Name of the column with p-value.")
    tool.add_argument("-a","--alpha",dest="alpha",action="store",type=float,
                        required=False,default=0.05, help="Alpha value.")
    # Tool Output
    output = parser.add_argument_group(title='Output paths', 
                        description="Paths for the output files")
    output.add_argument("-on","--outadjusted",dest="outadjusted",action="store",
                        required=True,help="Output path for corrected file[TSV]")
    output.add_argument("-fl","--flags",dest="flags",action="store",
                        required=True,help="Output path for flags file[TSV]")
    args = parser.parse_args()

    # Standardize paths
    args.input       = os.path.abspath(args.input)
    args.flags       = os.path.abspath(args.flags)
    args.outadjusted = os.path.abspath(args.outadjusted)

    return (args)

def main(args):
    # Reading input data, set "rowID" as index and just the pval column
    logger.info("Importing data")
    toCorrect_df = pd.read_csv(args.input, sep="\t")
    toCorrect_df.set_index(args.uniqID, inplace=True)
    justPvals = toCorrect_df[args.pval].values

    # Making bonferroni, Benjamini/Hochberg, Benjamini/Yekutieli
    # Alpha for FWER, family-wise error rate, e.g. 0.1
    # http://statsmodels.sourceforge.net/devel/generated/statsmodels.sandbox.stats.multicomp.multipletests.html
    logger.info("Runnig corrections")
    bonferroni = pd.Series(stm.multipletests(justPvals,alpha=args.alpha,
                    returnsorted=False, method="bonferroni")[1],
                    name=args.pval+"_bonferroni", index=toCorrect_df.index)
    bHochberg  = pd.Series(stm.multipletests(justPvals,alpha=args.alpha,
                    returnsorted=False, method="fdr_bh")[1],
                    name=args.pval+"_bHochberg", index=toCorrect_df.index)
    bYekutieli = pd.Series(stm.multipletests(justPvals,alpha=args.alpha,
                    returnsorted=False, method="fdr_by")[1],
                    name=args.pval+"_bYekutieli", index=toCorrect_df.index)

    # Creating objet with flags
    # Add a column for each correction
    logger.info("Getting Flags")
    significance_flags = Flags(index=toCorrect_df.index)
    for test in [bonferroni,bHochberg,bYekutieli]:
        significance_flags.addColumn(column="flag_{0}_significant".format(test.name),
                                     mask=(test<args.alpha))
    
    # Concatenating results with pvals
    results = pd.concat([toCorrect_df[args.pval],bonferroni,bHochberg,bYekutieli],axis=1)

    # Saving data
    logger.info("Saving results and flags")
    results.to_csv(args.outadjusted,sep="\t")
    significance_flags.df_flags.to_csv(args.flags,sep="\t")
    logger.info("Script Complete!")

if __name__ == '__main__':
    # Import data
    args = getOptions()

    # Setting logger
    logger = logging.getLogger()
    sl.setLogger(logger)
    logger.info("Importing data with following parameters:"\
                "\n\tInput: {0}"\
                "\n\tuniqID: {1}"\
                "\n\tpVal: {2}".format(args.input, args.uniqID, args.pval))

    # Start main
    main(args)
