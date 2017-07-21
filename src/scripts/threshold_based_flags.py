#!/usr/bin/env python
################################################################################
# DATE: 2017/03/09
#
# SCRIPT: threshold_based_flags.py
#
# VERSION: 2.2
# 
# AUTHOR: Coded by: Miguel A Ibarra (miguelib@ufl.edu) 
# 
# DESCRIPTION: This script flags features based on a given threshold.
#
################################################################################
# Import built-in libraries
import os
import logging
import argparse

# Import local data libraries
from secimtools.dataManager import logger as sl
from secimtools.dataManager.flags import Flags
from secimtools.dataManager.interface import wideToDesign

def getOptions():
    """Function to pull in arguments"""
    description = """The tool generates flags for features with values above the pre-specified threshold."""
    parser = argparse.ArgumentParser(description=description)
    # Standard Input
    standard = parser.add_argument_group(title='Standard input', 
            description='Standard input for SECIM tools.')
    standard.add_argument("-i", "--input", dest="input", action='store', 
                        required=True, help="Input dataset in wide format.")
    standard.add_argument("-d", "--design", dest="design", action='store', 
                        required=True, help="Design file.")
    standard.add_argument("-id", "--ID", dest="uniqID", action='store', 
                        required=True, help="Name of the column with unique:"\
                        " identifiers.")
    standard.add_argument("-g", "--group", dest="group", action='store', 
                        required=True, default=None, help="Add the option to "\
                        "separate sample IDs by treatement name. ")
    # Tool Input
    tool = parser.add_argument_group(title='Tool input', 
            description='Tool specific input.')
    tool.add_argument("-c", "--cutoff", dest="cutoff", action='store', 
                        required=False, default=30000, type=int, 
                        help="Cutoff to use for which values to flag. This "\
                        "defaults to 30,000")
    # Tool Output
    output = parser.add_argument_group(title='Tool output', 
            description='Tool output paths.')
    output.add_argument("-o", "--output", dest="output", action='store', 
                        required=True, help="Output path for the created "\
                        "flag file.")
    args = parser.parse_args()

    # Standardize paths
    args.input  = os.path.abspath(args.input)
    args.design = os.path.abspath(args.design)
    args.output = os.path.abspath(args.output)

    return(args)


def main(args):
    # Import data
    logger.info("Importing data with the interface")
    dat = wideToDesign(args.input, args.design, args.uniqID)

    # Cleaning from missing data
    dat.dropMissing()
    
    # Iterate through each group to add flags for if a group has over half of
    # its data above the cutoff
    logger.info("Running threshold based flags")
    df_offFlags = Flags(index=dat.wide.index)
    for title, group in dat.design.groupby(args.group):
        mask   = (dat.wide[group.index] < args.cutoff)
        meanOn = mask.mean(axis=1)
        df_offFlags.addColumn(column='flag_feature_' + title + '_off', 
                              mask=meanOn > 0.5)

    logger.info("Creating output")
    df_offFlags.df_flags.to_csv(args.output, sep="\t")

if __name__ == "__main__":
    # Command line options
    args = getOptions()

    # Setting logger
    logger = logging.getLogger()
    sl.setLogger(logger)

    # Starting script
    logger.info("Importing data with following parameters:"\
                "\n\tInput: {0}"\
                "\n\tDesign: {1}"\
                "\n\tuniqID: {2}"\
                "\n\tgroup: {3}".format(args.input, args.design, args.uniqID, 
                                args.group))
    # Plotting import info 
    logger.info('Importing Data')
    main(args)
