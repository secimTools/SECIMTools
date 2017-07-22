#!/usr/bin/env python
################################################################################
# DATE: 2017/06/13
# 
# MODULE: data_normalization_and_rescalling.py
#
# VERSION: 2.0
# 
# AUTHORS: Miguel Ibarra (miguelib@ufl.edu) and 
#          Alexander Kirpich (akirpich@ufl.edu)
#
# DESCRIPTION: Normalizations/transformations of the data based on
# "sum", "median", "mean", "centering", "pareto", "auto", "range", "level" and "vast" scaling.
#
################################################################################
# Import built-in libraries
import os
import logging
import argparse

# Import add-on libraries
import numpy as np
import pandas as pd

# Import local data libraries
from secimtools.dataManager import logger as sl
from secimtools.dataManager.interface import wideToDesign

def getOptions():
    """Function to pull arguments"""
    parser = argparse.ArgumentParser(description="The tool takes data and performs normalization/re-scaling" \
                                                 "based on the method selected by the user.")
    # Standar Input
    standar = parser.add_argument_group(title='Standard input', 
                description='Standard input for SECIM tools.')
    standar.add_argument("-i","--input",dest="input", action='store', 
                        required=True, help="Input dataset in wide format.")
    standar.add_argument("-d","--design",dest="design", action='store', 
                        required=True, help="Design file.")
    standar.add_argument("-id","--uniqID",dest="uniqID",action="store",
                        required=True, help="Name of the column with unique." \
                        "identifiers.")
    # Tool Input
    tool = parser.add_argument_group(title='Tool specific input', 
                description='Input specific for this tool.')
    tool.add_argument("-m","--method", dest="method", action='store', 
                        required=True, choices=["mean", "sum", "median", "centering", "auto", "range", "pareto", "level", "vast" ], 
                        help="Name of the normalization method that user wants to apply.")
    # Tool output
    output = parser.add_argument_group(title='Output paths', 
                description="Paths for the output files")
    output.add_argument("-o","--out",dest="out",action="store",
                        required=True,help="Path for TSV output of the normalized/re-scalled data.")
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

    # Cleaning from missing data
    dat.dropMissing()

    # Transpose data to normalize
    toNormalize_df =  dat.wide.T



    # Telling the user about the selected normalization method.

    logger.info("Normalizing data using {0} method.".format(args.method))


    # mean, median and sum are applied per sample across features!!!!
    if args.method == "mean" or args.method == "sum" or args.method == "median":      
 
       if args.method == "mean":
          toNormalize_df[args.method] = toNormalize_df.mean(axis=1)
          logger.info("Mean scaling is used for each sample across features.")
     
       if args.method == "sum":
          toNormalize_df[args.method] = toNormalize_df.sum(axis=1)
          logger.info("Sum scaling is used for each sample across features.")
    
       if args.method == "median":
          toNormalize_df[args.method] = toNormalize_df.median(axis=1)
          logger.info("Median scaling is used for each sample across features.")
       
       # Dividing by factor
       toNormalize_df = toNormalize_df.apply(lambda x: x/x[args.method], axis=1)


       # Dropping extra column
       toNormalize_df.drop(args.method, axis=1, inplace=True)

 
    # "centering", "auto", "range", "pareto", "level", "vast" are performed per feature across samples!!!!
    else:      

        # Computing mean for each feature.
        feature_value_means = toNormalize_df.mean(axis=0)

        if args.method == "centering":
        
           # Performing centering for each feature. 
           # In this case the value fo each feature will have mean zero across samples.
           logger.info("Centering is used for each feature across samples.")
           toNormalize_df = toNormalize_df - feature_value_means


        if args.method == "auto":
        
           # Computing standard deviation for each feature.
           feature_value_std = toNormalize_df.std(axis=0, ddof=1)
        
           # Performing auto-sclaing. 
           # In this case the value fo each feature will have mean zero and std = 1 across samples.
           logger.info("Autoscaling is used for each feature across samples.")
           toNormalize_df = (toNormalize_df - feature_value_means)/feature_value_std


        if args.method == "pareto":

           # Computing standard deviation and the square root of it for each feature.
           feature_value_std = toNormalize_df.std(axis=0, ddof=1)
           feature_value_std_sqrt = np.sqrt(feature_value_std)
       
           # Performing Pareto Scaling. The only difference from auto-scaling is that we use sqrt(standar_deviation).
           # In this case the value fo each feature will have mean zero and std will NOT be 1 across samples.
           logger.info("Pareto scaling is used for each feature across samples.")
           toNormalize_df = (toNormalize_df - feature_value_means)/feature_value_std_sqrt


        if args.method == "range":

           # Computing mean, min, max and range for each feature.
           feature_value_min = toNormalize_df.min(axis=0)
           feature_value_max = toNormalize_df.max(axis=0)
           feature_value_max_min = feature_value_max - feature_value_min        

           # Performing range scaling. Each feature is centered and divided by the range of that feature.
           logger.info("Range scaling is used for each feature across samples.")
           toNormalize_df = (toNormalize_df - feature_value_means)/feature_value_max_min


        if args.method == "level":

           # Performing level scaling. Each feature is centered and divided by the the mean of that feature.
           logger.info("Level scaling is used for each feature across samples.")
           toNormalize_df = (toNormalize_df - feature_value_means)/feature_value_means


        if args.method == "vast":

           # Computing standard deviation and coefficient of variaiton for each feature.
           feature_value_std = toNormalize_df.std(axis=0, ddof=1)
           feature_value_cv = feature_value_std/feature_value_means

           # Performing VarianceStabilizing (VAST) scaling. Each feature is centered and divided by the coefficient of variation.
           logger.info("VAST scaling is used for each feature across samples.")
           toNormalize_df = (toNormalize_df - feature_value_means)/feature_value_std
           toNormalize_df =  toNormalize_df/feature_value_cv
    

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
