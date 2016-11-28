#!/usr/bin/env python
################################################################################
# DATE: 2016/October/10
# 
# MODULE: subsetData.py
#
# VERSION: 1.1
# 
# AUTHOR: Miguel Ibarra (miguelib@ufl.edu) 
#
# DESCRIPTION: Subsets wide data based on groups or any other field on the
#               design file. 
#
################################################################################

#Standard Libraries
import os
import logging
import argparse
from itertools import chain

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
    standar = parser.add_argument_group(title='Standard input', description= 
                                        'Standard input for SECIM tools.')
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
    standar.add_argument("-dp","--drops", dest="drops", action='store', 
                        required=True, help="Name of the groups in your"\
                        "group/treatment column that you want to keep.")

    #Output Paths
    output = parser.add_argument_group(title='Output paths', description=
                                       "Paths for the output files")
    #output.add_argument("-od","--outdesign",dest="outdesign",action="store",
    #                    required=True,help="Output path for flags file[CSV]")
    output.add_argument("-ow","--outwide",dest="outwide",action="store",
                        required=True,help="Output path for bff file[CSV]")

    args = parser.parse_args()
    return (args)

def main(args):
    # Importing data trough
    dat = wideToDesign(args.input, args.design, args.uniqID, group=args.group)

    # Treating data as numeric
    dat.wide = dat.wide.applymap(float)

    # Spliting keepers
    drops = args.drops.split(",")

    # Making sure all the groups to drop actually exist on the design column
    if args.group:
        for todrop in drops:
            if todrop in dat.levels:
                pass
            else:
                logger.error("Your group '{0}' is not located in the column '{1}'".
                    format(todrop,dat.group))
                # This may not be the best way to end the script
                exit()

    # If the subsetting is going to be made by group the select de sampleIDs 
    # from the design file
    logger.info(u"Getting sampleNames to drop")
    if args.group:
        iToDrop=list()
        for name,group in dat.design.groupby(dat.group):
            if name in drops:
                iToDrop+=(group.index.tolist())
    else:
        iToDrop=drops

    # Dropping elements
    #for name,group in dat.design.groupby(dat.group) :
    #    if name in drops:
    #        print group.index.tolist()
    selectedWide = dat.wide.drop(iToDrop, axis=1, inplace=False)

    # Output wide results
    logger.info(u"""Output wide file.""")
    selectedWide.to_csv(os.path.abspath(args.outwide), sep='\t')


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
                group: {3}
                ToDrop: {4}
                """.format(args.input, args.design, args.uniqID, args.group,
                    args.drops))

    # Main script
    main(args)