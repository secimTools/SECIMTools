## Goal ##
# need a script that can merge all of your flag files together by index
# Add logic to check if the indexs dont match and throw an error if they do not


# parser.add_argument('--foo', nargs="+") '+' means one or more while '*' means any number
# Add try catch and documentation

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
    # Need to take each arg and turn into data frame and add to new list
    flagDataFrameList = []
    logger.info("Importing data")

    # Check for commas, commas are used in galaxy. If there are commas separate the list by commas
    if ',' in args.flagFiles[0]:
        args.flagFiles = args.flagFiles[0].split(',')

    for flagFile in args.flagFiles:
        dataFrame = pd.DataFrame.from_csv(flagFile, sep='\t')
        flagDataFrameList.append(dataFrame)

    logger.info("Checking all indexes are the same")
    counter = 0
    while counter < len(flagDataFrameList) - 1:
        if flagDataFrameList[counter].index.equals(flagDataFrameList[counter + 1].index):
            counter += 1
        else:
            logger.error("Not all indexes the same")
            counter += 1

    mergedFlags = pd.concat(flagDataFrameList, axis=1)
    # mergedFlags.to_csv(args.mergedFile, sep='\t')
    print mergedFlags.head()
def main(args):
    mergeFlags(args)


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    sl.setLogger(logger)
    main(args)
