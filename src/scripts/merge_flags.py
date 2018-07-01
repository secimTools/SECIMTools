#!/usr/bin/env python
################################################################################
# DATE: 2016/May/06, rev: 2016/July/11
#
# SCRIPT: merge_flags.py
#
# VERSION: 2.0
# 
# AUTHOR: Jonathan Poisson(poissonj@ufl.edu) Miguel A Ibarra (miguelib@ufl.edu)
# 
# DESCRIPTION: The script merges files containing flas.
#
################################################################################
# Import built-in libraries
import os
import re
import logging
import argparse

# Import add-on libraries
import pandas as pd

# Import local data libraries
from secimtools.dataManager import logger as sl
from secimtools.dataManager.flags import Flags
from secimtools.dataManager.interface import wideToDesign


def getOptions():
    """ Function to pull in arguments """
    parser = argparse.ArgumentParser()
    # Tool Input
    tool = parser.add_argument_group(title='Tool input')
    tool.add_argument("-i","--input", dest="flagFiles", action='store', 
                        required=True, nargs="+", help="Input any number of "\
                        "flag files that have the same indexes")
    tool.add_argument("-f","--filename", dest="filename", action='store', 
                        required=True, nargs="+", help="Filename for input data.")
    tool.add_argument("-fid","--flagUniqID",dest="flagUniqID",action="store",
                        required=False, default="rowID",help="Name of the column "\
                        "with unique identifiers in the flag files.")
    # Tool Output
    output = parser.add_argument_group(title='Required output')
    output.add_argument("-o", "--output", dest="mergedFile", action='store', 
                        required=True, help="Output file")
    args = parser.parse_args()

    # Standardize paths
    args.mergedFile = os.path.abspath(args.mergedFile)

    return args

def cleanStr(x):
    """ 
    Clean strings so they behave.

    :Arguments:
        x (str): A string that needs cleaning

    :Returns:
        x (str): The cleaned string.

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
    # Need to take each arg and turn into data frame and add to new list
    flagDataFrameList = []
    logger.info("Importing data")
    if ',' in args.flagFiles[0]:
        args.flagFiles = args.flagFiles[0].split(',')
    print(args.flagFiles)
    if args.filename:
        filenames = [cleanStr(x=fname) for fname in args.filename]
    print(filenames)
    for flagFile,filename in zip(args.flagFiles,filenames):
        dataFrame = pd.read_table(flagFile)
        if args.flagUniqID:
            try:
                dataFrame.set_index(args.flagUniqID, inplace=True)
            except:
                logger.error("Index {0} does not exist on file.".format(args.flagUniqID))
        dataFrame.columns=[name+"_"+filename for name in dataFrame.columns]
        flagDataFrameList.append(dataFrame)
    mergedFlags = Flags.merge(flagDataFrameList)
    # NOTE: Pandas cannot store NANs as an int. If there are NANs from the
    # merge, then the column becomes a float. Here I change the float output to
    # look like an int.
    mergedFlags.to_csv(args.mergedFile, float_format='%.0f', sep='\t')
    logger.info("Script Complete!")


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    # Setting logger
    logger = logging.getLogger()
    sl.setLogger(logger)

    # Running main script
    main(args)

