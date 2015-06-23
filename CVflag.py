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
from matplotlib.backends.backend_pdf import PdfPages
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
    parser.add_argument("--log", dest = "log", action = 'store_true', required = False, default = False, help = "Put this argument if the data needs rounding to 3 significant digits and a log transformation (Usually for peak area and peak height). [option]")
    parser.add_argument("--debug", dest = "debug", action = 'store_true', required = False, help = "Add debugging log output. [optional]")
    parser.add_argument("--group", dest = "group", action = 'store', required = False, default = False, help = "Add option to separate sample IDs by treatment name. [optional]")
    #parser.add_argument("--html", dest = "html", action = 'store', required = False, default = False, help = "Html file outut name. Do not use for cammand line use. [optional]")
    parser.add_argument("--outPath", dest = "outPath", action = 'store', required = True, help = "Path to save created files.")
    parser.add_argument("--CVflag", dest = "outfile", action = 'store', required = False, default = 'CVflag', help = "Output filename of CV flags. The default is [CVflag]. [optional]")
    #parser.add_argument("--CVtype", dest = "CVtype", action = 'store', required = False, default = False, help = "Only for the use of flag name. E.g. if you put PA, the name of the flag is flag_PA_big_CV. If leave it empty, the name is flag_big_CV as default. [optional]")
    parser.add_argument("--CVcutoff", dest = "CVcutoff", action = 'store', required = False, default = False, help = "The default CV cutoff will flag 10 percent of the rowIDs with larger CVs. If you want to set a CV cutoff, put the number here. [optional]")
    parser.add_argument("--CVplot", dest = "CVplot", action = 'store', required = False, default = 'CVplot', help = "Output filename of CV plot. The default is [CVplot]. [optional]")

    if myopts:
        args = parser.parse_args(myopts)
    else:
        args = parser.parse_args()

    return(args)

"""class FileName:
    def __init__(self, text, fileType, groupName=''):
        self.text = str(text)   # Convert all to string in case of numbers as names
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
            self.fileNameWithSlash = '/' + self.groupName + '_' + self.text + self.fileType"""

def setCVflagByGroup(args, wide, dat, dir):

    # Split design file by treatment group
    
    try:
        pdfOut = PdfPages(dir + '/' + args.CVplot + '_by_' + args.group + '.pdf')
        CV = pd.DataFrame(index=wide.index)
        for title, group in dat.design.groupby(args.group):

            # Filter the wide file into a new dataframe
            currentFrame = wide[group.index]

            # Change dat.sampleIDs to match the design file
            dat.sampleIDs = group.index

            CV['cv_'+title], CVcutoff = setCVflag(args, currentFrame, dat, dir=dir, groupName=title)
            
        CV['cv'] = CV.apply(np.max, axis=1)
        if not args.CVcutoff:
            CVcutoff = np.percentile(CV['cv'].values, q=90)
            CVcutoff = round(CVcutoff, -int(floor(log(CVcutoff, 10))) + 2)
        else:
            CVcutoff = args.CVcutoff
        for title, group in dat.design.groupby(args.group):
        
            fig, ax = plt.subplots()
            xmin = -np.percentile(CV['cv_'+title].values,99)*0.2
            xmax = np.percentile(CV['cv_'+title].values,99)*1.5
            ax.set_xlim(xmin, xmax)
            CV['cv_'+title].plot(kind='hist', range = (xmin, xmax), bins = 15, normed = 1, color = 'grey', label = "CV histogram")
            CV['cv_'+title].plot(kind='kde', title="Density Plot of Coefficients of Variation in " + args.group + " " + title, ax=ax, label = "CV density")
            plt.axvline(x=CVcutoff, color = 'red', linestyle = 'dashed', label = "Cutoff at: {0}".format(CVcutoff))
            plt.legend()
            pdfOut.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        fig, ax = plt.subplots()
        xmin = -np.percentile(CV['cv'].values,99)*0.2
        xmax = np.percentile(CV['cv'].values,99)*1.5
        ax.set_xlim(xmin, xmax)
        for title, group in dat.design.groupby(args.group):
            CV['cv_'+title].plot(kind='kde', title="Density Plot of Coefficients of Variation by " + args.group, ax=ax, label = "CV density in group "+title)
        plt.axvline(x=CVcutoff, color = 'red', linestyle = 'dashed', label = "Cutoff at: {0}".format(CVcutoff))
        plt.legend()
        pdfOut.savefig(fig, bbox_inches='tight')
        plt.close(fig)    
        pdfOut.close()
        return CV['cv'], CVcutoff

    except KeyError as e:
        logger.error("{} is not a column name in the design file.".format(args.group))
    except Exception as e:
        logger.error("Error. {}".format(e))

def setCVflag(args, wide, dat, dir, groupName = ''):
    
    # Round all values to 3 significant digits
    DATround = wide.applymap(lambda x: x)
    if args.log:
        DATround = DATround.applymap(lambda x: log(round(x, -int(floor(log(abs(x), 10))) + 2)))
    
    # Get std, mean and calculate CV
    DATstat = pd.DataFrame(index=DATround.index)
    DATstat['std']  = DATround.apply(np.std, axis=1)
    DATstat['mean'] = DATround.apply(np.mean, axis=1)
    DATstat['cv']   = DATstat['std'] / DATstat['mean']
    
    if not args.CVcutoff:
        CVcutoff = np.percentile(DATstat['cv'].values, q=90)
        CVcutoff = round(CVcutoff, -int(floor(log(CVcutoff, 10))) + 2)
    else:
        CVcutoff = args.CVcutoff
    
    """
    # Set CV flags
    CVflag = pd.DataFrame(index=DATround.index)
    if not groupName == '':
        grName = '_' + groupName
    else:
        grName = ''
    if args.CVtype:
        CVtype = '_' + args.CVtype
    else:
        CVtype = ''
    flagName = 'flag' + CVtype + '_big_CV' + grName 
    CVflag[flagName] = DATstat['cv'] > CVcutoff
    CVflag = CVflag.applymap(lambda x: int(x))

    # Output flags
    CVflagFileName = args.outfile + '.tsv'
    CVflagFile = open(dir + '/' + CVflagFileName, 'w')
    CVflagFile.write(CVflag.to_csv(sep='\t'))
    CVflagFile.close()"""
    
    #if args.html:
    #    htmlContents.append('<li style=\"margin-bottom:1.5%;\"><a href="{}">{}</a></li>'.format(CVflagFileName.fileName, groupName + ' CVflag'))
        
    # Plot CVs
    if groupName == '':
        fig, ax = plt.subplots()
        #xmin, xmax = ax.get_xlim()
        xmin = -np.percentile(DATstat['cv'].values,99)*0.2
        xmax = np.percentile(DATstat['cv'].values,99)*1.5
        ax.set_xlim(xmin, xmax)
        DATstat['cv'].plot(kind='hist', range = (xmin, xmax), bins = 15, normed = 1, color = 'grey', ax = ax, label = "CV histogram")
        DATstat['cv'].plot(kind='kde', title="Density Plot of Coefficients of Variation", ax=ax, label = "CV density")
        plt.axvline(x=CVcutoff, color = 'red', linestyle = 'dashed', label = "Cutoff at: {0}".format(CVcutoff))
        plt.legend()
        CVplotFileName = args.CVplot + '.pdf'
        CVplotFileNameWithDir = dir + '/' + CVplotFileName
        plt.savefig(CVplotFileNameWithDir, format='pdf')
        plt.close(fig)
    
    return DATstat['cv'], CVcutoff

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
        htmlContents = ["<html><head><title>CV flag List</title></head><body>"]
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

    # Use group separation or not depending on user input
    if not args.group:
        DATstat, CVcutoff = setCVflag(args, wide, dat, dir = directory)
        CVflagFileName = args.outfile
    else:
        DATstat, CVcutoff = setCVflagByGroup(args, wide, dat, dir = directory)
        CVflagFileName = args.outfile + '_by_' + args.group
    CVflag = pd.DataFrame(index=DATstat.index)
    CVflag['flag_big_CV'] = DATstat > CVcutoff
    CVflag = CVflag.applymap(lambda x: int(x))
    CVflagFileNameDir = directory + '/' + CVflagFileName + '.tsv'
    CVflagFile = open(CVflagFileNameDir, 'w')
    #CVflagFile.write(DATstat.to_csv(sep='\t'))
    CVflag.to_csv(CVflagFileNameDir, sep='\t')
    CVflagFile.close()    

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


















