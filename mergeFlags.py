#!/usr/bin/env python

# Author: Jonathan Poisson | poissonj@ufl.edu

# Built-in packages
import argparse
import logging

# Add-on packages
import pandas as pd

# Local Packages
import logger as sl
from interface import Flags


def getOptions():
    """ Function to pull in arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="flagFiles", action='store', required=True, nargs="+",
                        help="Input any number of flag files that have the same indexes")
    parser.add_argument('--output', dest="mergedFile", action='store', required=True, help="Output file")

    args = parser.parse_args()
    return args


def mergeFlags(args):
    """
    :Arguments:
        :type args: argparse.ArgumentParser
        :param args: Command line arguments

    :Returns:
        :rtype: .tsv
        :returns: Merged flags tsv file
    """
    # Need to take each arg and turn into data frame and add to new list
    flagDataFrameList = []
    logger.info("Importing data")

    # Check for commas, commas are used in galaxy. If there are commas separate
    # the list by commas
    if ',' in args.flagFiles[0]:
        args.flagFiles = args.flagFiles[0].split(',')

    # Convert files into dataframes and populate into new list
    for flagFile in args.flagFiles:
        dataFrame = pd.DataFrame.from_csv(flagFile, sep='\t')
        flagDataFrameList.append(dataFrame)

    logger.info("Checking all indexes are the same")

    # Merge flags using Flags class
    mergedFlags = Flags.merge(flagDataFrameList)

    # Export merged flags
    # NOTE: Pandas cannot store NANs as an int. If there are NANs from the
    # merge, then the column becomes a float. Here I change the float output to
    # look like an int.
    mergedFlags.to_csv(args.mergedFile, float_format='%.0f', sep='\t')


def main(args):
    # Call mergeFlags function. Main is used for convention
    mergeFlags(args)


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    sl.setLogger(logger)
    main(args)

