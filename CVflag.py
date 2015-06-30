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
    #parser.add_argument("--log", dest = "log", action = 'store_true', required = False, default = False, help = "Put this argument if the data needs rounding to 3 significant digits and a log transformation (Usually for peak area and peak height). [option]")
    parser.add_argument("--debug", dest = "debug", action = 'store_true', required = False, help = "Add debugging log output. [optional]")
    parser.add_argument("--group", dest = "group", action = 'store', required = False, default = False, help = "Add option to separate sample IDs by treatment name. [optional]")
    parser.add_argument("--outPath", dest = "outPath", action = 'store', required = True, help = "Path to save created files.")
    #parser.add_argument("--CVflag", dest = "outfile", action = 'store', required = False, default = 'CVflag', help = "Output filename of CV flags. The default is [CVflag]. [optional]")
    parser.add_argument("--CVcutoff", dest = "CVcutoff", action = 'store', required = False, default = False, help = "The default CV cutoff will flag 10 percent of the rowIDs with larger CVs. If you want to set a CV cutoff, put the number here. [optional]")
    #parser.add_argument("--CVplot", dest = "CVplot", action = 'store', required = False, default = 'CVplot', help = "Output filename of CV plot. The default is [CVplot]. [optional]")

    if myopts:
        args = parser.parse_args(myopts)
    else:
        args = parser.parse_args()

    return(args)

def setCVflagByGroup(args, wide, dat, dir):

    # Split design file by treatment group
    
    try:
        pdfOut = PdfPages(dir + '/CVplot_by_' + args.group + '.pdf') # (dir + '/' + args.CVplot + '_by_' + args.group + '.pdf')
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
    #if args.log:
    #    DATround = DATround.applymap(lambda x: log(round(x, -int(floor(log(abs(x), 10))) + 2)))
    
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
        CVplotFileName = 'CVplot.pdf' # args.CVplot + '.pdf'
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
    
    # Import data
    dat = wideToDesign(args.fname, args.dname, args.uniqID)

    # Only interested in samples
    wide = dat.wide[dat.sampleIDs]

    # Use group separation or not depending on user input
    if not args.group:
        DATstat, CVcutoff = setCVflag(args, wide, dat, dir = directory)
        CVflagFileName = 'CVflag' # args.outfile
    else:
        DATstat, CVcutoff = setCVflagByGroup(args, wide, dat, dir = directory)
        CVflagFileName = 'CVflag_by_' + args.group # args.outfile + '_by_' + args.group
    CVflag = pd.DataFrame(index=DATstat.index)
    CVflag['flag_big_CV'] = DATstat > CVcutoff
    CVflag = CVflag.applymap(lambda x: int(x))
    CVflagFileNameDir = directory + '/' + CVflagFileName + '.tsv'
    CVflagFile = open(CVflagFileNameDir, 'w')
    CVflag.to_csv(CVflagFileNameDir, sep='\t')
    CVflagFile.close()    

if __name__ == '__main__':

    # Command line options
    args = getOptions()
    
    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
    else:
        sl.setLogger(logger)

    main(args)


















