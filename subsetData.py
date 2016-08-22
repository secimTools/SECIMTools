#!/usr/bin/env python
######################################################################################
# DATE: 2016/August/10
# 
# MODULE: subsetData.py
#
# VERSION: 1.0
# 
# AUTHOR: Miguel Ibarra (miguelib@ufl.edu) 
#
# DESCRIPTION: Selects the groups to use in further analysis. 
#
#######################################################################################

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
    standar.add_argument("-g","--group", dest="group", action='store', 
                        required=True, help="Name of column in design file" \
                        "with Group/treatment information.")
    standar.add_argument("-k","--keepers", dest="keepers", action='store', 
                        required=True, help="Name of the groups in your"\
                        "group/treatment column that you want to keep.")

    #Output Paths
    output = parser.add_argument_group(title='Output paths', description=
                                       "Paths for the output files")
    output.add_argument("-od","--outdesign",dest="outdesign",action="store",
                        required=True,help="Output path for flags file[CSV]")
    output.add_argument("-ow","--outwide",dest="outwide",action="store",
                        required=True,help="Output path for bff file[CSV]")

    args = parser.parse_args()
    return (args)

def main(args):
    # Importing data trough
    dat = wideToDesign(args.input, args.design, args.uniqID, args.group)

    # Treating data as numeric
    dat.wide = dat.wide.applymap(float)

    # Spliting keepers
    keepers = args.keepers.split(",")

    # Making sure all the groups to keep actually exist on the design column
    for keeper in keepers:
        if keeper in dat.levels:
            pass
        else:
            logger.error("Your group '{0}'' is not located in the column '{1}'".
                format(keeper,dat.group))
            exit()

    # If the groups to keep are located on the design the subset the wide
    logger.info(u"""Selected groups to keep""")
    selectedWide = [dat.wide[group.index] for name,group in  \
                    dat.design.groupby(dat.group) if name in keepers]

    # If the groups to keep are located on the design the subset the design
    selectedDesing = [group for name,group in  dat.design.groupby(dat.group) \
                    if name in keepers]

    # Makin design file complete again
    selectedDesing =  pd.concat(selectedDesing)

    # Makin wide file complete again
    selectedWide = pd.concat(selectedWide, axis=1)

    # Outputing results to final files.
    logger.info(u"""Output wide file.""")
    selectedDesing.to_csv(os.path.abspath(args.outdesign), sep='\t')

    logger.info(u"""Output design file.""")
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
                keepers: {4}
                """.format(args.input, args.design, args.uniqID, args.group,
                    args.keepers))
    main(args)