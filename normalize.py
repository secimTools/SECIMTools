#!/usr/bin/env python
################################################################################
# DATE: 2016/August/23
# 
# MODULE: subsetData.py
#
# VERSION: 1.0
# 
# AUTHOR: Miguel Ibarra (miguelib@ufl.edu) 
#
# DESCRIPTION: Selects the groups to use in further analysis. 
#
################################################################################

#Standard Libraries
import os
import logging
import argparse

#AddOn Libraries
import numpy as np
import pandas as pd

# Local Packages
import logger as sl
from interface import wideToDesign

def getOptions():
    """Function to pull arguments"""
    parser = argparse.ArgumentParser(description="Takes a peak area/heigh \
                                     dataset and calculates the LOD on it ")

    #Standar input for SECIMtools
    standar = parser.add_argument_group(title='Standard input', description= 
                                        'Standard input for SECIM tools.')
    standar.add_argument("-i","--input",dest="input", action='store', 
                        required=True, help="Input dataset in wide format.")
    standar.add_argument("-d","--design",dest="design", action='store', 
                        required=True, help="Design file.")
    standar.add_argument("-id","--uniqID",dest="uniqID",action="store",
                        required=True, help="Name of the column with unique" \
                        "dentifiers.")
    standar.add_argument("-m","--method", dest="method", action='store', 
                        required=True, choices=["mean","sum"], help="Name of the groups "\
                        "in your group/treatment column that you want to keep.")

    #Output Paths
    output = parser.add_argument_group(title='Output paths', description=
                                       "Paths for the output files")
    output.add_argument("-on","--outnormalized",dest="outnormalized",action="store",
                        required=True,help="Output path for flags file[TSV]")

    args = parser.parse_args()
    return (args)

def main(args):
    # Importing data trough
    dat = wideToDesign(args.input, args.design, args.uniqID)

    # Treating data as numeric
    dat.wide = dat.wide.applymap(float)

    # Transpose data to normalize
    toNormalize_df =  dat.wide.T

    logger.info("Normalizing data using {0} method".format(args.method))
    if args.method == "mean":
        # Getting the mean
        toNormalize_df["mean"] = toNormalize_df.mean(axis=1)

        # Dividing by the mean
        toNormalize_df = toNormalize_df.apply(lambda x: x/x["mean"], axis=1)

        # Dropping extra column
        toNormalize_df.drop("mean", axis=1, inplace=True)


    elif args.method == "sum":
        # Getting the sum
        toNormalize_df["sum"] = toNormalize_df.sum(axis=1)

        # Dividing by the mean
        toNormalize_df = toNormalize_df.apply(lambda x: x/x["sum"], axis=1)

        # Dropping extra column
        toNormalize_df.drop("sum", axis=1, inplace=True)


    # Transposing normalized data
    normalized_df = toNormalize_df.T

    # Saving data
    normalized_df.to_csv(os.path.abspath(args.outnormalized),sep="\t")

if __name__ == '__main__':
    #Import data
    args = getOptions()

    #Setting logger
    logger = logging.getLogger()
    sl.setLogger(logger)
    logger.info(u"""Importing data with following parameters:
                Input: {0}
                Design: {1}
                uniqID: {2}
                method: {3}
                """.format(args.input, args.design, args.uniqID, args.method))
    main(args)