# Author: Jonathan Poisson | poissonj@ufl.edu

# Built-in packages
import argparse
import logging

# Add-on packages
import pandas as pd

# Local Packages
import logger as sl

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
    Arguments:
        :type args: argparse.ArgumentParser
        :param args: Command line arguments

    Returns:
        :rtype: .tsv
        :returns: Merged flags tsv file
    """
    # Need to take each arg and turn into data frame and add to new list
    flagDataFrameList = []
    logger.info("Importing data")

    # Check for commas, commas are used in galaxy. If there are commas separate the list by commas
    if ',' in args.flagFiles[0]:
        args.flagFiles = args.flagFiles[0].split(',')

    # Convert files into dataframes and populate into new list
    for flagFile in args.flagFiles:
        dataFrame = pd.DataFrame.from_csv(flagFile, sep='\t')
        flagDataFrameList.append(dataFrame)

    logger.info("Checking all indexes are the same")
    counter = 0
    try:
        while counter < len(flagDataFrameList) - 1:
            # Check the index of a dataframe compared to the rest of the dataframe's indexes
            if flagDataFrameList[counter].index.equals(flagDataFrameList[counter + 1].index):
                counter += 1
            else:
                raise IOError
    except IOError:
        logger.error("Not all indexes are the same")
        raise SystemExit

    # Merge flags together
    mergedFlags = pd.concat(flagDataFrameList, axis=1)
    # Export merged flags
    mergedFlags.to_csv(args.mergedFile, sep='\t')

def main(args):
    # Call mergeFlags function. Main is used for convention
    mergeFlags(args)


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    sl.setLogger(logger)
    main(args)
