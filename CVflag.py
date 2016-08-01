#!/usr/bin/env python

# Built-in packages
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


def getOptions(myopts=None):
    """Function to pull in arguments"""
    
    description = """The coefficient of variation (CV) is defined as the ratio
    of the sample standard deviation to the mean. It is a method to measure the
    variations of compounds. The variation of a peak intensity increases as 
    its CV increases. And adjusted for the sample mean, CV does not have unit;
    thus, it is a standardized measurement for variation. 
    
    A density plot of CVs for all compounds across samples by group is performed.
    And a set of flags of compounds with large CVs will be output. """
    
    parser = argparse.ArgumentParser(description = description, formatter_class = RawDescriptionHelpFormatter)

    group1 = parser.add_argument_group(title='Standard input', description='Standard input for SECIM tools.')
    group1.add_argument("--input", dest = "fname", action = 'store', required = True, help = "Input dataset in wide format.")
    group1.add_argument("--design", dest = "dname", action = 'store', required = True, help = "Design file.")
    group1.add_argument("--ID", dest = "uniqID", action = 'store', required = True, help = "Name of the column with unique identifiers.")
    group1.add_argument("--CVplotOutFile", dest = "CVplot", action = 'store', required = True, default = 'CVplot', help = "Name of the output PDF for CV plots.")
    group1.add_argument("--CVflagOutFile", dest = "CVflag", action = 'store', required = True, default = 'RTflag', help = "Name of the output TSV for CV flags.")

    group2 = parser.add_argument_group(title='Optional input', description='Optional input for SECIM tools.')
    group2.add_argument("--group", dest = "group", action = 'store', required = False, default = False, help = "Add option to separate sample IDs by treatment name. [optional]")
    group2.add_argument("--CVcutoff", dest = "CVcutoff", action = 'store', required = False, default = False, help = "The default CV cutoff will flag 10 percent of the rowIDs with larger CVs. If you want to set a CV cutoff, put the number here. [optional]")

    args = parser.parse_args()

    return(args)

def galaxySavefig(fig, fname):
    """ Take galaxy DAT file and save as fig """

    png_out = fname + '.png'
    fig.savefig(png_out)

    # shuffle it back and clean up
    data = file(png_out, 'rb').read()
    with open(fname, 'wb') as fp:
        fp.write(data)
    os.remove(png_out)


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
        CVcutoff = np.nanpercentile(CV['cv'].values, q=90)
        CVcutoff = round(CVcutoff, -int(floor(log(abs(CVcutoff), 10))) + 2)
    else:
        CVcutoff = float(args.CVcutoff)
    for title, group in dat.design.groupby(args.group):
        fig, ax = plt.subplots()
        xmin = -np.nanpercentile(CV['cv_'+title].values,99)*0.2
        xmax = np.nanpercentile(CV['cv_'+title].values,99)*1.5
        ax.set_xlim(xmin, xmax)
        CV['cv_'+title].plot(kind='hist', range = (xmin, xmax), bins = 15, normed = 1, color = 'grey', label = "CV histogram")
        CV['cv_'+title].plot(kind='kde', title="Density Plot of Coefficients of Variation in " + args.group + " " + title, ax=ax, label = "CV density")
        plt.axvline(x=CVcutoff, color = 'red', linestyle = 'dashed', label = "Cutoff at: {0}".format(CVcutoff))
        plt.legend()
        pdfOut.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots()
    xmin = -np.nanpercentile(CV['cv'].values,99)*0.2
    xmax = np.nanpercentile(CV['cv'].values,99)*1.5
    ax.set_xlim(xmin, xmax)

    # Create flag file instance
    CVflag = Flags(index=CV['cv'].index)

    for title, group in dat.design.groupby(args.group):
        CV['cv_'+title].plot(kind='kde', title="Density Plot of Coefficients of Variation by " + args.group, ax=ax, label = "CV density in group "+title)

        # Create new flag row for each group
        CVflag.addColumn(column='flag_feature_big_CV_' + title,
                     mask=((CV['cv_'+title].get_values() > CVcutoff) | CV['cv_'+title].isnull()))

    plt.axvline(x=CVcutoff, color = 'red', linestyle = 'dashed', label = "Cutoff at: {0}".format(CVcutoff))
    plt.legend()
    pdfOut.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    pdfOut.close()

    # Write flag file
    CVflag.df_flags.to_csv(args.CVflag, sep='\t')


def setCVflag(args, wide, dat, groupName = ''):

    # Round all values to 3 significant digits
    DATround = wide.applymap(lambda x: x)

    # Get std, mean and calculate CV #
    DATstat = pd.DataFrame(index=DATround.index)

    DATstat['std']  = DATround.apply(np.std, axis=1)
    DATstat['mean'] = DATround.apply(np.mean, axis=1)
    DATstat['cv']   = abs(DATstat['std'] / DATstat['mean'])

    if not args.CVcutoff:
        CVcutoff = np.nanpercentile(DATstat['cv'].values, q=90)
        CVcutoff = round(CVcutoff, -int(floor(log(abs(CVcutoff), 10))) + 2)
    else:
        CVcutoff = float(args.CVcutoff)

    # Plot CVs
    if groupName == '':
        fig, ax = plt.subplots()
        xmin = -np.nanpercentile(DATstat['cv'].values,99)*0.2
        xmax = np.nanpercentile(DATstat['cv'].values,99)*1.5
        ax.set_xlim(xmin, xmax)
        DATstat['cv'].plot(kind='hist', range = (xmin, xmax), bins = 15, normed = 1, color = 'grey', ax = ax, label = "CV histogram")
        DATstat['cv'].plot(kind='kde', title="Density Plot of Coefficients of Variation", ax=ax, label = "CV density")
        plt.axvline(x=CVcutoff, color = 'red', linestyle = 'dashed', label = "Cutoff at: {0}".format(CVcutoff))
        plt.legend()

        # Set file name of pdf and export
        #CVplotFileName = args.CVplot
        plt.savefig(args.CVplot, format='pdf')
        plt.close(fig)

        # Create flag instance
        CVflag = Flags(index=DATstat.index)

        # Create new flag column with flags
        CVflag.addColumn(column='flag_feature_big_CV',
                     mask=((DATstat['cv'].get_values() > CVcutoff) | DATstat['cv'].isnull()))

        # Write output
        CVflag.df_flags.to_csv(args.CVflag, sep='\t')
    else:
        return DATstat['cv'], CVcutoff


def main(args):

    # Import data
    dat = wideToDesign(args.fname, args.dname, args.uniqID)

    # Only interested in samples
    wide = dat.wide[dat.sampleIDs]

    # Use group separation or not depending on user input
    if not args.group:
        setCVflag(args, wide, dat)
    else:
        setCVflagByGroup(args, wide, dat)


if __name__ == '__main__':

    # Command line options
    args = getOptions()

    main(args)

