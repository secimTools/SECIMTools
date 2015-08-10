#!/usr/bin/env python

# Built-in packages
import logging
import argparse
import shutil
import os

# Add-on packages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Local Packages
from interface import wideToDesign
from interface import Flags
import logger as sl


def getOptions(myopts=None):
    """ Function to pull in arguments """
    description = """ Count the digits in data to determine possible outliers
                      or discrepancies"""
    parser = argparse.ArgumentParser(description=description)

    requiredInput = parser.add_argument_group(description="Required input")
    requiredInput.add_argument("--input", dest="fname", action='store', required=True, help="Input dataset in wide format.")
    requiredInput.add_argument("--design", dest="dname", action='store', required=True, help="Design file.")
    requiredInput.add_argument("--ID", dest="uniqID", action='store', required=True, help="Name of the column with unique identifiers.")
    requiredInput.add_argument("--html_path", dest="htmlPath", action='store', required=True, help="Path to save created files and html file")
    requiredInput.add_argument("--html", dest="html", action='store', required=True, help="Html file output name")
    requiredInput.add_argument("--flags", dest="flags", action='store', required=True, help="Flag file output")

    optionalInput = parser.add_argument_group(description="Optional input")
    optionalInput.add_argument("--noZero", dest="zero", action='store_true', required=False, help="Flag to ignore zeros.")
    optionalInput.add_argument("--debug", dest="debug", action='store_true', required=False, help="Add debugging log output.")
    optionalInput.add_argument("--group", dest="group", action='store', required=False, default=False, help="Add the option to separate sample IDs by treatement name. ")
    optionalInput.add_argument("--noZip", dest="noZip", action='store_true', required=False, default=False, help="If running from command line use --noZip to skip the zip creation. This stops the command line from freezing")

    if myopts:
        args = parser.parse_args(myopts)
    else:
        args = parser.parse_args()

    return(args)


def splitDigit(x):
    """ Function to split digits by decimal

        :Arguments:
            :param integer x: Digit to split

        :Returns:
            :rtype: integer
            :returns: integer that is split by the decimal

    """

    if x == 0:
        cnt = np.nan
    else:
        # Bug fixer of scientific notation (Very large and very small numbers)
        x = str('%f' % x)

        # Split x at the decimal point and then take the length of the string
        # Before the decimal point
        cnt = len(x.split('.')[0])
    return cnt


class FileName:
    """ Class to create a file name for accurate location in the Galaxy file system """
    def __init__(self, text, fileType, groupName=''):
        """ Constructor

            :Arguments:
                :param string text: Text of the file name
                :param string fileType: '.tsv', '.txt', '.csv'
                :param string groupName: If using groups to separate data, the group name will be added to the file name
        """
        self.text = str(text)  # Must convert all to string just in case of numbers as names
        self.fileType = str(fileType)
        self.groupName = str(groupName)
        self._createFileName()
        self._createFileNameWithSlash()

    def _createFileName(self):
        if self.groupName == '':
            self.fileName = self.text + self.fileType
        else:
            self.fileName = self.groupName + '_' + self.text + self.fileType

    def _createFileNameWithSlash(self):
        if self.groupName == '':
            self.fileNameWithSlash = '/' + self.text + self.fileType
        else:
            self.fileNameWithSlash = '/' + self.groupName + '_' + self.text + self.fileType


def countDigitsByGroups(args, wide, dat, dir):
    """ If the group option is selected this function is called to split by groups.

    The function calls the countDigits function in a loop that iterates through
    the groups

        :Arguments:
            :type args: argparse.ArgumentParser
            :param args: Command line arguments

            :type wide: pandas.DataFrame
            :param wide: A data frame in wide format

            :type dat: pandas.DataFrame
            :param dat: A data frame in design format

            :param string dir: String of the directory name for storing files in galaxy

    """
    # Split Design file by group
    try:
        for title, group in dat.design.groupby(args.group):

            # Filter the wide file into a new dataframe
            currentFrame = wide[group.index]

            # Change dat.sampleIDs to match the design file
            dat.sampleIDs = group.index

            countDigits(currentFrame, dat, dir=dir, groupName=title)
    except KeyError:
        logger.error("{} is not a column name in the design file.".format(args.group))
    except Exception as e:
        logger.error("Error. {}".format(e))


def countDigits(wide, dat, dir, groupName=''):
    """ Function to create and export the counts, figure, and summary files

        :Arguments:
            :type wide: pandas.DataFrame
            :param wide: A data frame in wide format

            :type dat: pandas.DataFrame
            :param dat: A data frame in design format

            :param string dir: String of the directory name for storing files in galaxy

            :param string groupName: Name of the group if using the group option. Set to an empty stirng by default

    """
    # Count the number of digits before decimal and get basic distribution info
    cnt = wide.applymap(lambda x: splitDigit(x))
    cnt['min'] = cnt.apply(np.min, axis=1)
    cnt['max'] = cnt.apply(np.max, axis=1)
    cnt['diff'] = cnt['max'] - cnt['min']

    # Create mask of differences. If the difference is greater than 1 a flag needs to be made
    mask = cnt['diff'] >= 2

    # Update the global flag file with mask
    flag.update(mask)

    # write output
    cntFileName = FileName(text='counts', fileType='.tsv', groupName=groupName)
    cntFile = open(dir + cntFileName.fileNameWithSlash, 'w')
    cntFile.write(cnt.to_csv(sep='\t'))
    cntFile.close()

    htmlContents.append('<li style=\"margin-bottom:1.5%;\"><a href="{}">{}</a></li>'.format(cntFileName.fileName, groupName + ' Counts'))

    # Make distribution plot of differences

    # Set title to default if there is none
    if groupName:  # If a title is set (set a different title for each group and export everything in a zip)
        title = 'Distribution of difference between min and max across compounds: ' + str(groupName)
        # shutil.make_archive("CountDigitsArchive", "zip")  # This creates a zip of everything, not good

    else:  # groups are not being used
        title = 'Distribution of difference between min and max across compounds'

    if cnt['diff'].any():
        fig, ax = plt.subplots(figsize=(8, 8))
        cnt['diff'].plot(kind='hist', ax=ax, title=title)
        ax.set_xlabel('Difference in Number of Digits (max - min)')
        ax.set_ylabel('Number of Features')

        # Save figure into archive
        figureFileName = FileName(text='difference(Max-Min)', fileType='.png', groupName=groupName)
        fig.savefig(dir + figureFileName.fileNameWithSlash, bbox_inches='tight')
        htmlContents.append('<li style=\"margin-bottom:1.5%;\"><a href="{}">{}</a></li>'.format(figureFileName.fileName, groupName + ' Figure'))
    else:
        logger.warn('There were no differences in digit counts, no plot will be generated')


def main(args):
    # Create a directory in galaxy to hold the files created
    directory = args.htmlPath
    try:  # for test - needs this done
        os.makedirs(args.htmlPath)
    except Exception, e:
        logger.error("Error. {}".format(e))

    htmlFile = file(args.html, 'w')

    global htmlContents
    # universe_wsgi.ini file's html_sanitizing must be false to allow for styling
    htmlContents = ["<html><head><title>Count Digits Results List</title></head><body>"]
    htmlContents.append('<div style=\"background-color:black; color:white; text-align:center; margin-bottom:5% padding:4px;\">'
                        '<h1>Output</h1>'
                        '</div>')
    htmlContents.append('<ul style=\"text-align:left; margin-left:5%;\">')

    # Import data
    logger.info(u'html system path: {}'.format(args.htmlPath))
    logger.info(u'Importing data with following parameters: \n\tWide: {0}\n\tDesign: {1}\n\tUnique ID: {2}'.format(args.fname, args.dname, args.uniqID))
    dat = wideToDesign(args.fname, args.dname, args.uniqID)

    # Only interested in samples
    wide = dat.wide[dat.sampleIDs]

    # Global flag file
    global flag
    flag = Flags(index=wide.index)
    flag.addColumn(column='flag_feature_count_digits')

    # Use group separation or not depending on user input
    if args.group:
        countDigitsByGroups(args, wide, dat, dir=directory)
    else:
        countDigits(wide, dat, dir=directory)

    # Create a zip archive with the inputted zip file name of the temp file
    if args.noZip:
        pass
    else:
        shutil.make_archive(directory + '/Archive_of_Results', 'zip', directory)

    # Add zip of all the files to the list
    htmlContents.append('<li><a href="{}">{}</a></li>'.format('Archive_of_Results.zip', 'Zip of Results'))

    # Close html list and contents
    htmlContents.append('</ul></body></html>')

    htmlFile.write("\n".join(htmlContents))
    htmlFile.write("\n")
    htmlFile.close()

    # Output flag file
    flag.df_flags.to_csv(args.flags, sep="\t")


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
    else:
        sl.setLogger(logger)

    main(args)

