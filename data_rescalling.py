#!/usr/bin/env python
################################################################################
# DATE: 2016/October/4
# 
# MODULE: subsetData.py
#
# VERSION: 1.1
# 
# AUTHOR: Miguel Ibarra (miguelib@ufl.edu) 
#
# DESCRIPTION: Normalizes data based on sum, median or mean.
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
    parser = argparse.ArgumentParser(description="Takes a peak area/heigh" \
                                     "dataset and calculates the LOD on it ")

    #Standar input for SECIMtools
    standar = parser.add_argument_group(title='Standard input', 
                description='Standard input for SECIM tools.')
    standar.add_argument("-i","--input",dest="input", action='store', 
                        required=True, help="Input dataset in wide format.")
    standar.add_argument("-d","--design",dest="design", action='store', 
                        required=True, help="Design file.")
    standar.add_argument("-id","--uniqID",dest="uniqID",action="store",
                        required=True, help="Name of the column with unique" \
                        "dentifiers.")

    tool = parser.add_argument_group(title='Tool specific input', 
                description='Input specific for this tool.')
    tool.add_argument("-m","--method", dest="method", action='store', 
                        required=True, choices=["mean","sum","median"], 
                        help="Name of the groups in your group/treatment column"\
                        " that you want to keep.")

    #Output Paths
    output = parser.add_argument_group(title='Output paths', 
                description="Paths for the output files")
    output.add_argument("-o","--out",dest="out",action="store",
                        required=True,help="Output path for flags file[TSV]")
    args = parser.parse_args()

    # Stadardize paths
    args.out    = os.path.abspath(args.out)
    args.input  = os.path.abspath(args.input)
    args.design = os.path.abspath(args.design)

    return (args)

def main(args):
    # Importing data trough
    logger.info("Loading data trough the interface")
    dat = wideToDesign(args.input, args.design, args.uniqID, logger=logger)

    # Treating data as numeric
    dat.wide = dat.wide.applymap(float)

    # Transpose data to normalize
    toNormalize_df =  dat.wide.T

    # Selecting method for normalization
    logger.info("Normalizing data using {0} method".format(args.method))
    if args.method == "mean":
        toNormalize_df[args.method] = toNormalize_df.mean(axis=1)
    elif args.method == "sum":
        toNormalize_df[args.method] = toNormalize_df.sum(axis=1)
    elif args.method == "median":
        toNormalize_df[args.method] = toNormalize_df.median(axis=1)

    # Dividing by factor
    toNormalize_df = toNormalize_df.apply(lambda x: x/x[args.method], axis=1)

    # Dropping extra column
    toNormalize_df.drop(args.method, axis=1, inplace=True)

    # Transposing normalized data
    normalized_df = toNormalize_df.T

    # Saving data
    normalized_df.to_csv(args.out,sep="\t")
    logger.info("Script Complete!")

if __name__ == '__main__':
    #Import data
    args = getOptions()

    #Setting logger
    logger = logging.getLogger()
    sl.setLogger(logger)
    logger.info("Importing data with following parameters:"\
                "\n\tInput: {0}"\
                "\n\tDesign: {1}"\
                "\n\tuniqID: {2}"\
                "\n\tmethod: {3}".format(args.input, args.design,
                 args.uniqID, args.method))

    # Starting script
    main(args)