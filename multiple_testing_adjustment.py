#!/usr/bin/env python
################################################################################
# DATE: 2016/August/23
# 
# MODULE: adjustPval.py
#
# VERSION: 1.0
# 
# AUTHOR: Miguel Ibarra (miguelib@ufl.edu) 
#
# DESCRIPTION: Adjust pvalues. 
#
################################################################################

#Standard Libraries
import os
import logging
import argparse

#AddOn Libraries
import numpy as np
import pandas as pd
import statsmodels.sandbox.stats.multicomp as stm

# Local Packages
import logger as sl
from interface import wideToDesign
from flags import Flags

def getOptions():
    """Function to pull arguments"""
    parser = argparse.ArgumentParser(description="Takes a peak area/heigh" \
                                    " dataset and calculates the LOD on it ")

    #Standar input for SECIMtools
    standar = parser.add_argument_group(title='Standard input', description= 
                                        'Standard input for SECIM tools.')
    standar.add_argument("-i","--input",dest="input", action='store', 
                        required=True, help="Input dataset in wide format.")
    standar.add_argument("-id","--uniqID",dest="uniqID",action="store",
                        required=True, help="Name of the column with unique" \
                        "dentifiers.")
    standar.add_argument("-pv","--pval",dest="pval",action="store",
                        required=True, help="Name of the column with p-value.")
    standar.add_argument("-a","--alpha",dest="alpha",action="store",type=float,
                        required=False,default=0.05, help="Alpha value.")

    #Output Paths
    output = parser.add_argument_group(title='Output paths', description=
                                       "Paths for the output files")
    output.add_argument("-on","--outadjusted",dest="outadjusted",action="store",
                        required=True,help="Output path for corrected file[TSV]")
    output.add_argument("-fl","--flags",dest="flags",action="store",
                        required=True,help="Output path for flags file[TSV]")

    args = parser.parse_args()
    return (args)

def main(args):
    # Importing data trough
    toCorrect_df = pd.read_csv(args.input, sep="\t")

    # Set rowID as indexes
    toCorrect_df.set_index(args.uniqID, inplace=True)

    # select just pvals
    justPvals = toCorrect_df[args.pval].values

    # Making bonferroni, Benjamini/Hochberg, Benjamini/Yekutieli
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

    #Creating objet with flags
    significance_flags = Flags(index=toCorrect_df.index)

    #Add a column for each correction
    for test in [bonferroni,bHochberg,bYekutieli]:
        significance_flags.addColumn(column="flag_{0}_off".format(test.name),
            mask=(test>args.alpha))
    
    # Concatenating results with pvals
    results = pd.concat([toCorrect_df[args.pval],bonferroni,bHochberg,bYekutieli],axis=1)

    # Saving data
    logger.info("Saving results")
    results.to_csv(os.path.abspath(args.outadjusted),sep="\t")

    logger.info("Saving flags")
    significance_flags.df_flags.to_csv(os.path.abspath(args.flags),sep="\t")

if __name__ == '__main__':
    # Import data
    args = getOptions()

    # Setting logger
    logger = logging.getLogger()
    sl.setLogger(logger)
    logger.info(u"""Importing data with following parameters:
                Input: {0}
                uniqID: {1}
                pVal: {2}
                """.format(args.input, args.uniqID, args.pval))

    # Start main
    main(args)