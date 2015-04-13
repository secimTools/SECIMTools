#!/usr/bin/env python

# Built-in packages
import logging
import argparse
from argparse import RawDescriptionHelpFormatter

# Add-on packages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local Packages
from interface import wideToDesign
import logger as sl


def getOptions():
    """ Function to pull in arguments """
    description = """ One-Way ANOVA """
    parser = argparse.ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--input", dest="fname", action='store', required=True, help="Input dataset in wide format.")
    parser.add_argument("--design", dest="dname", action='store', required=True, help="Design file.")
    parser.add_argument("--ID", dest="uniqID", action='store', required=True, help="Name of the column with unique identifiers.")
    parser.add_argument("--noZero", dest="zero", action='store_true', required=False, help="Flag to ignore zeros.")
    parser.add_argument("--out", dest="oname", action='store', required=True, help="Output file name for counts table.")
    parser.add_argument("--summary", dest="sname", action='store', required=True, help="Output file name for summary table.")
    parser.add_argument("--fig", dest="figname", action='store', required=True, help="Name of output figure name.")
    parser.add_argument("--debug", dest="debug", action='store_true', required=False, help="Add debugging log output.")
    args = parser.parse_args()
#     args = parser.parse_args(['--input', '/home/jfear/sandbox/secim/data/ST000015.tsv',
#                               '--design', '/home/jfear/sandbox/secim/data/ST000015_design.tsv',
#                               '--ID', 'Name',
#                               '--out', '/home/jfear/sandbox/secim/data/test.csv',
#                               '--summary', '/home/jfear/sandbox/secim/data/test_summary.csv',
#                               '--fig', '/home/jfear/sandbox/secim/data/test.png',
#                               '--debug'])
    return(args)


def splitDigit(x):
    """ Function to split digits by decimal """
    if x == 0:
        cnt = np.nan
    else:
        cnt = len(str(x).split('.')[0])
    return cnt


def main(args):
    """ """
    # Import data
    logger.info(u'Importing data with following parameters: \n\tWide: {0}\n\tDesign: {1}\n\tUnique ID: {2}'.format(args.fname, args.dname, args.uniqID))
    dat = wideToDesign(args.fname, args.dname, args.uniqID)

    # Only interested in samples
    wide = dat.wide[dat.sampleIDs]

    # Count the number of digits before decimal and get basic distribution info
    cnt = wide.applymap(lambda x: splitDigit(x))
    cnt['min'] = cnt.apply(np.min, axis=1)
    cnt['max'] = cnt.apply(np.max, axis=1)
    cnt['diff'] = cnt['max'] - cnt['min']
    cnt['argMax'] = cnt[dat.sampleIDs].apply(np.argmax, axis=1)
    cnt['argMin'] = cnt[dat.sampleIDs].apply(np.argmin, axis=1)

    # wite output
    cnt.to_csv(args.oname)

    # Make distribution plot of differences
    fig, ax = plt.subplots(figsize=(8, 8))
    cnt['diff'].plot(kind='kde', ax=ax, title='Distribution of difference between min and max across compounds')
    ax.set_xlabel('Difference (max - min)')
    plt.tight_layout()
    plt.savefig(args.figname)

    # Summarize the number of times a smaple had the most of fewest digits
    maxSample = cnt['argMax'].value_counts()
    minSample = cnt['argMin'].value_counts()
    maxSample.name = 'max_num'
    minSample.name = 'min_num'
    summary = pd.concat((maxSample, minSample), axis=1)
    summary.fillna(0, inplace=True)
    summary.sort(columns='min_num', inplace=True, ascending=False)
    summary[['max_num', 'min_num']] = summary[['max_num', 'min_num']].astype(int)

    # wite output
    summary.to_csv(args.sname)


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
    else:
        sl.setLogger(logger)

    main(args)
