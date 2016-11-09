#!/usr/bin/env python

# Author: Jonathan Poisson | poissonj@ufl.edu
# Eddited by:  Miguel ibarra | miguelib@ufl.edu

# Built-in packages
import re
import argparse
import logging

# Add-on packages
import pandas as pd

# Local Packages
import logger as sl
from flags import Flags


def getOptions():
    """ Function to pull in arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="flagFiles", action='store', required=True, nargs="+",
                        help="Input any number of flag files that have the same indexes")
    parser.add_argument("--filename", dest="filename", action='store', required=True, nargs="+",
                        help="Filename for input data.")
    parser.add_argument('-fid',"--flagUniqID",dest="flagUniqID",action="store",
                    required=False, default="rowID",help="Name of the column "\
                    "with unique identifiers in the flag files.")
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

    # If args.filename is provided then use it to add its name to column names
    # This paramether will should be used only on galaxy
    if args.filename:
        # Cleaning weird characters on file names and replacing them with '_'. 
        filenames = [cleanStr(x=fname) for fname in args.filename]


    # Convert files into dataframes and populate into new list
    for flagFile,filename in zip(args.flagFiles,filenames):
        # Read table
        dataFrame = pd.read_table(flagFile)

        # Flag uniqID
        if args.flagUniqID:
            try:
                dataFrame.set_index(args.flagUniqID, inplace=True)
            except:
                logger.error("Index {0} does not exist on file.".format(args.flagUniqID))

        dataFrame.columns=[name+"_"+filename for name in dataFrame.columns]

        # List of frame
        flagDataFrameList.append(dataFrame)

    #logger.info("Checking all indexes are the same")

    # Merge flags using Flags class
    mergedFlags = Flags.merge(flagDataFrameList)

    # Export merged flags
    # NOTE: Pandas cannot store NANs as an int. If there are NANs from the
    # merge, then the column becomes a float. Here I change the float output to
    # look like an int.
    mergedFlags.to_csv(args.mergedFile, float_format='%.0f', sep='\t')

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
    # Call mergeFlags function. Main is used for convention
    mergeFlags(args=args)


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    sl.setLogger(logger)
    main(args)

