#!/usr/bin/env python

# Built-in packages
import logging
import argparse
from argparse import RawDescriptionHelpFormatter
import tempfile
import shutil
import os
from math import log, floor

# Add-on packages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local packages
from interface import wideToDesign
import logger as sl

# Glocals
global DEBUG
DEBUG = False

def getOptions(myopts=None):
    """Function to pull in arguments"""
    description = """Retention Time flags are ... """
    parser = argparse.ArgumentParser(description = description, formatter_class = RawDescriptionHelpFormatter)
    parser.add_argument("--input", dest = "fname", action = 'store', required = True, help = "Input dataset in wide format.")
    parser.add_argument("--design", dest = "dname", action = 'store', required = True, help = "Design file.")
    parser.add_argument("--ID", dest = "uniqID", action = 'store', required = True, help = "Name of the column with unique identifiers.")
    parser.add_argument("--debug", dest = "debug", action = 'store_true', required = False, help = "Add debugging log output. [optional]")
    #parser.add_argument("--group", dest = "group", action = 'store', required = False, default = False, help = "Add option to separate sample IDs by treatment name. [optional]")
    parser.add_argument("--pctl", dest = "p90p10", action = 'store_true', required = False, default = False, help = "The difference is calculated by 95th percentile and 5th percentile by default. If you add this argument, it uses 90th percentile and 10th percentile. [optional]")
    #parser.add_argument("--html", dest = "html", action = 'store', required = False, default = False, help = "Html file outut name. Do not use for cammand line use. [optional]")
    parser.add_argument("--outPath", dest = "outPath", action = 'store', required = True, help = "Path to save output files")
    parser.add_argument("--RTflagOutFile", dest = "outfile", action = 'store', required = False, default = 'RTflag', help = "Output filename of RT flags. The default is [RTflag]. [optional]")
    parser.add_argument("--CVcutoff", dest = "CVcutoff", action = 'store', required = False, default = False, help = "The default CV cutoff will flag 10 percent of the rowIDs with larger CVs. If you want to set a CV cutoff, put the number here. [optional]")
    parser.add_argument("--CVplotOutFile", dest = "CVplot", action = 'store', required = False, default = 'CVplot', help = "Output filename of CV plot. The default is [CVplot]. [optional]")

    if myopts:
        args = parser.parse_args(myopts)
    else:
        args = parser.parse_args()

    return(args)

"""class FileName:
    def __init__(self, text, fileType, groupName=''):
        self.text = str(text)   # Convert all to string in case of numbers as names
        self.fileType = str(fileType)
        #self.groupName = str(groupName)
        self._createFileName()
        self._createFileNameWithSlash()
        
    def _createFileName(self):
        #if self.groupName == '':
            self.fileName = self.text + self.fileType
        #else:
        #    self.fileName = self.groupName + '_' + self.text + self.fileType
            
    def _createFileNameWithSlash(self):
        if self.groupName == '':
            self.fileNameWithSlash = '/' + self.text + self.fileType
        else:
            self.fileNameWithSlash = '/' + self.groupName + '_' + self.text + self.fileType"""
"""
def setRTflagByGroup(args, wide, dat, dir):

    # Split design file by treatment group
    try:
        for title, group in dat.design.groupby(args.group):

            # Filter the wide file into a new dataframe
            currentFrame = wide[group.index]

            # Change dat.sampleIDs to match the design file
            dat.sampleIDs = group.index

            setRTflag(args, currentFrame, dat, dir=dir, groupName=title)

    except KeyError as e:
        logger.error("{} is not a column name in the design file.".format(args.group))
    except Exception as e:
        logger.error("Error. {}".format(e))"""

def setRTflag(args, wide, dat, dir):
    
    # Round retention time to 2 decimals
    RTround = wide.applymap(lambda x: round(x, 2))
    
    # Get percentiles, min, max, mean, median
    RTstat = pd.DataFrame(index=RTround.index)
    RTstat['min'] = RTround.apply(np.min, axis=1)
    RTstat['max'] = RTround.apply(np.max, axis=1)
    RTstat['p95'] = RTround.apply(np.percentile, q=95, axis=1)
    RTstat['p90'] = RTround.apply(np.percentile, q=90, axis=1)
    RTstat['p10'] = RTround.apply(np.percentile, q=10, axis=1)
    RTstat['p05'] = RTround.apply(np.percentile, q= 5, axis=1)
    RTstat['std'] = RTround.apply(np.std, axis=1)
    RTstat['mean']   = RTround.apply(np.mean, axis=1)
    RTstat['median'] = RTround.apply(np.median, axis=1)
    RTstat['cv']     = RTstat['std'] / RTstat['mean']
    RTstat['p95p05'] = RTstat['p95'] - RTstat['p05']
    RTstat['p90p10'] = RTstat['p90'] - RTstat['p10']

    # Set RT flags
    RTflag = pd.DataFrame(index=RTround.index)
    if args.p90p10:
        RTflag['flag_RT_Q90Q10_outlier'] = (RTstat['p90p10'] > 0.2)
    else:
        RTflag['flag_RT_Q95Q05_outlier'] = (RTstat['p95p05'] > 0.2)
    RTflag['flag_RT_max_gt_threshold'] = (RTstat['max'] - RTstat['median'] > 0.1)
    RTflag['flag_RT_min_lt_threshold'] = (RTstat['min'] - RTstat['median'] < -0.1)
    RTflag['flag_RT_min_max_outlier'] = ((RTstat['max']-RTstat['mean']>3*RTstat['std']) | (RTstat['min']-RTstat['mean']<-3*RTstat['std']))
    if not args.CVcutoff:
        CVcutoff = np.percentile(RTstat['cv'].values, q=90)
        CVcutoff = round(CVcutoff, -int(floor(log(CVcutoff, 10))) + 2)
    else:
        CVcutoff = args.CVcutoff
    RTflag['flag_RT_big_CV'] = (RTstat['cv'] > CVcutoff)
    RTflag = RTflag.applymap(lambda x: int(x))

    # Output flags
    RTflagFileName = args.outfile + '.tsv'
    RTflagFile = open(dir + '/' + RTflagFileName, 'w')
    RTflagFile.write(RTflag.to_csv(sep='\t'))
    RTflagFile.close()
    """if args.html:
        htmlContents.append('<li style=\"margin-bottom:1.5%;\"><a href="{}">{}</a></li>'.format(RTflagFileName.fileName, groupName + ' RTflag'))"""
        
    # Plot RT CVs
    fig, ax = plt.subplots()
    #xmin, xmax = ax.get_xlim()
    xmin = -np.percentile(RTstat['cv'].values,99)*0.2
    xmax = np.percentile(RTstat['cv'].values,99)*1.5
    ax.set_xlim(xmin, xmax)
    #plt.hist(RTstat['cv'], range = (xmin, xmax), bins = 15, normed = 1, color = 'grey')
    RTstat['cv'].plot(kind='hist', range = (xmin, xmax), bins = 15, normed = 1, color = 'grey', ax=ax, label = "CV histogram")
    RTstat['cv'].plot(kind='kde', title="Density Plot of Coefficients of Variation of the Retention Time", ax=ax, label = "CV density")
    plt.axvline(x=CVcutoff, color = 'red', linestyle = 'dashed', label = "Cutoff at: {0}".format(CVcutoff))
    plt.legend()
    CVplotFileName = args.CVplot + '.pdf'
    CVplotFileNameWithDir = dir + '/' + CVplotFileName
    plt.savefig(CVplotFileNameWithDir, format='pdf')
    plt.close(fig)

def main(args):
    """ """
    directory = args.outPath
    # create a directory in galaxy to hold the files created
    try: # Copied from countDigits.py
        os.makedirs(args.outPath)
    except Exception, e:
        logger.error("Error. {}".format(e))

    """if args.html:
        htmlFile = file(args.html, 'w')

        global htmlContents
        # universe_wsgi.ini file's html_sanitizing must be false to allow for styling
        htmlContents = ["<html><head><title>RT flag List</title></head><body>"]
        htmlContents.append('<div style=\"background-color:black; color:white; text-align:center; margin-bottom:5% padding:4px;\">'
                        '<h1>Output</h1>'
                        '</div>')
        htmlContents.append('<ul style=\"text-align:left; margin-left:5%;\">')"""
    
    # Import data
    #logger.info(u'html system path: {}'.format(args.outPath))
    #logger.info(u'Importing data with following parameters: \n\tWide: {0}\n\tDesign: {1}\n\tUnique ID: {2}'.format(args.fname, args.dname, args.uniqID))
    dat = wideToDesign(args.fname, args.dname, args.uniqID)

    # Only interested in samples
    wide = dat.wide[dat.sampleIDs]

    # Set RT flags
    setRTflag(args, wide, dat, dir = directory)

    """if args.html:
        # Create a zip archive with the inputted zip file name of the temp file
        shutil.make_archive(directory + '/Archive_of_Results', 'zip', directory)

        # Add zip of all the files to the list
        htmlContents.append('<li><a href="{}">{}</a></li>'.format('Archive_of_Results.zip', 'Zip of Results'))

        # Close html list and contents
        htmlContents.append('</ul></body></html>')

        htmlFile.write("\n".join(htmlContents))
        htmlFile.write("\n")
        htmlFile.close()"""

if __name__ == '__main__':

    # Command line options
    args = getOptions()
    
    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
    else:
        sl.setLogger(logger)

    main(args)


















