#!/usr/bin/env python
################################################################################
# DATE: 2016/October/10
# 
# MODULE: subset_design.py
#
# VERSION: 1.2
# 
# AUTHOR: Miguel Ibarra (miguelib@ufl.edu) 
#
# DESCRIPTION: Subsets design file data based on groups or any other field on the
#               design file. 
#
################################################################################

# Standard Libraries
import os
import re
import logging
import argparse
from itertools import chain

# AddOn Libraries
import numpy as np
import pandas as pd

# Local Libraries
from secimtools.dataManager import logger as sl
from secimtools.dataManager.interface import wideToDesign

def getOptions():
    """Function to pull arguments"""
    parser = argparse.ArgumentParser(description="Takes a peak area/heigh" \
                                     "dataset and calculates the LOD on it ")
    # Standar Input
    standar = parser.add_argument_group(title="Standard input", 
                        description= "Standard input for SECIM tools.")
    standar.add_argument("-i","--input",dest="input", action='store', 
                        required=True, help="Input dataset in wide format.")
    standar.add_argument("-d","--design",dest="design", action='store', 
                        required=True, help="Design file.")
    standar.add_argument("-id","--uniqID",dest="uniqID",action="store",
                        required=True, help="Name of the column with unique" \
                        "dentifiers.")
    standar.add_argument("-g","--group", dest="group", action='store', 
                        required=False, help="Name of column in design file" \
                        "with Group/treatment information.")
    # Tool Especific
    tool = parser.add_argument_group(title="Tool specific input", 
                        description= "Input that is specific for this tool.")
    tool.add_argument("-dp","--drops", dest="drops", action='store', 
                        required=True, help="Name of the groups in your"\
                        "group/treatment column that you want to keep.")

    # Output Paths
    output = parser.add_argument_group(title='Output paths', 
                        description="Paths for the output files")
    output.add_argument("-o","--out",dest="out",action="store",
                        required=True,help="Output path for bff file[CSV]")
    args = parser.parse_args()

    # Standardize paths
    args.out    = os.path.abspath(args.out)
    args.input  = os.path.abspath(args.input)
    args.design = os.path.abspath(args.design)

    # Split groups/samples to drop
    args.drops = args.drops.split(",")

    return (args)

def cleanStr(x):
    """ Clean strings so they behave.

    For some modules, uniqIDs and groups cannot contain spaces, '-', '*',
    '/', '+', or '()'. For example, statsmodel parses the strings and interprets
    them in the model.

    :Arguments:
        x (str): A string that needs cleaning

    :Returns:
        x (str): The cleaned string.

        self.origString (dict): A dictionary where the key is the new
            string and the value is the original string. This will be useful
            for reverting back to original values.

    """
    if isinstance(x, str):
        val = x
        x = re.sub(r'^-([0-9].*)', r'__\1', x)
        x = x.replace(' ', '_')
        x = x.replace('.', '_')
        x = x.replace('-', '_')
        x = x.replace('*', '_')
        x = x.replace('/', '_')
        x = x.replace('+', '_')
        x = x.replace('(', '_')
        x = x.replace(')', '_')
        x = x.replace('[', '_')
        x = x.replace(']', '_')
        x = x.replace('{', '_')
        x = x.replace('}', '_')
        x = x.replace('"', '_')
        x = x.replace('\'', '_')
        x = re.sub(r'^([0-9].*)', r'_\1', x)
    return x

def main(args):
    # Importing data trough
    logger.info("Importing data through wideToDesign data manager")
    dat = wideToDesign(args.input, args.design, args.uniqID, 
                        logger=logger)

    # Cleaning from missing data
    dat.dropMissing()

    # Making sure all the groups to drop actually exist on the design column
    if args.group:
        for todrop in args.drops:
            if todrop in list(set(dat.design[args.group].values)):
                pass
            else:
                logger.error("The group '{0}' is not located in the column '{1}' "\
                            "of your design file".format(todrop,args.group))
                raise ValueError

    # If the subsetting is going to be made by group the select de sampleIDs 
    # from the design file
    logger.info(u"Getting sampleNames to drop")
    if args.group:
        iToDrop = list()
        for name,group in dat.design.groupby(args.group):
            if name in args.drops:
                iToDrop+=(group.index.tolist())
    else:
        iToDrop = args.drops

    # Remove weird characters
    iToDrop = [cleanStr(x) for x in iToDrop]

    # Dropping elements
    selectedDesign = dat.design.drop(iToDrop,axis=0, inplace=False)

    # Output wide results
    logger.info("Output wide file")
    selectedDesign.to_csv(args.out, sep='\t')
    logger.info("Script Complete!")


if __name__ == '__main__':
    #Import data
    args = getOptions()

    #Setting logger
    logger = logging.getLogger()
    sl.setLogger(logger)
    logger.info("Importing data with following parameters:"
                "\n\tInput: {0}"
                "\n\tDesign: {1}"
                "\n\tuniqID: {2}"
                "\n\tgroup: {3}"
                "\n\tToDrop: {4}".format(args.input, args.design, args.uniqID,
                 args.group, args.drops))

    # Main script
    main(args)
