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
from interface import Flags

# Glocals
global DEBUG
DEBUG = False

def getOptions(myopts=None):
    """Function to pull in arguments"""
    description = """Retention Time flags are ... """
    parser = argparse.ArgumentParser(description = description, formatter_class = RawDescriptionHelpFormatter)

    group1 = parser.add_argument_group(title='Standard input', description='Standard input for SECIM tools.')
    group1.add_argument("--input", dest = "fname", action = 'store', required = True, help = "Input dataset in wide format.")
    group1.add_argument("--design", dest = "dname", action = 'store', required = True, help = "Design file.")
    group1.add_argument("--ID", dest = "uniqID", action = 'store', required = True, help = "Name of the column with unique identifiers.")
    #group1.add_argument("--outPath", dest = "outPath", action = 'store', required = True, help = "Path to save created files. Example: ~/Desktop/output/  The output folder must have been created already.")
    group1.add_argument("--CVplotOutFile", dest = "CVplot", action = 'store', required = True, default = 'CVplot', help = "Name of the output PDF for CV plots.")
    group1.add_argument("--CVflagOutFile", dest = "CVflag", action = 'store', required = True, default = 'RTflag', help = "Name of the output TSV for CV flags.")

    group2 = parser.add_argument_group(title='Standard input', description='Standard input for SECIM tools.')
    group2.add_argument("--group", dest = "group", action = 'store', required = False, default = False, help = "Add option to separate sample IDs by treatment name. [optional]")
    group2.add_argument("--CVcutoff", dest = "CVcutoff", action = 'store', required = False, default = False, help = "The default CV cutoff will flag 10 percent of the rowIDs with larger CVs. If you want to set a CV cutoff, put the number here. [optional]")

    args = parser.parse_args()

    return(args)

def setCVflagByGroup(args, wide, dat):

    # Split design file by treatment group

    pdfOut = PdfPages(args.CVplot)
    CV = pd.DataFrame(index=wide.index)
    for title, group in dat.design.groupby(args.group):

        # Filter the wide file into a new dataframe
        currentFrame = wide[group.index]

        # Change dat.sampleIDs to match the design file
        dat.sampleIDs = group.index

        CV['cv_'+title], CVcutoff = setCVflag(args, currentFrame, dat, groupName=title)

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


def setCVflag(args, wide, dat, groupName = ''):

    # Round all values to 3 significant digits
    DATround = wide.applymap(lambda x: x)

    # Get std, mean and calculate CV #
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
        xmin = -np.percentile(DATstat['cv'].values,99)*0.2
        xmax = np.percentile(DATstat['cv'].values,99)*1.5
        ax.set_xlim(xmin, xmax)
        DATstat['cv'].plot(kind='hist', range = (xmin, xmax), bins = 15, normed = 1, color = 'grey', ax = ax, label = "CV histogram")
        DATstat['cv'].plot(kind='kde', title="Density Plot of Coefficients of Variation", ax=ax, label = "CV density")
        plt.axvline(x=CVcutoff, color = 'red', linestyle = 'dashed', label = "Cutoff at: {0}".format(CVcutoff))
        plt.legend()

        # Set file name of pdf and export
        #CVplotFileName = args.CVplot
        plt.savefig(args.CVplot, format='pdf')
        plt.close(fig)

    return DATstat['cv'], CVcutoff

def main(args):

    # Import data
    dat = wideToDesign(args.fname, args.dname, args.uniqID)

    # Only interested in samples
    wide = dat.wide[dat.sampleIDs]

    # Use group separation or not depending on user input
    if not args.group:
        DATstat, CVcutoff = setCVflag(args, wide, dat)
        #CVflagFileName = 'CVflag'
    else:
        DATstat, CVcutoff = setCVflagByGroup(args, wide, dat)
        #CVflagFileName = 'CVflag_by_' + args.group

    #Create CVflag DataFrame
    CVflag = Flags(index=DATstat.index)

    # Fill in DataFrame values
    CVflag.addColumn(column = 'flag_big_CV', mask = DATstat > CVcutoff)

    #Set the file name and export
    CVflag.df_flags.to_csv(args.CVflag, sep='\t')

if __name__ == '__main__':

    # Command line options
    args = getOptions()

    main(args)

