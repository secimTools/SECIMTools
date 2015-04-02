#!/usr/bin/env python

# Built-in packages
import logging
import argparse
from argparse import RawDescriptionHelpFormatter

# Add-on packages
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpld3


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
    parser.add_argument("--group", dest="group", action='store', required=False, help="Group/treatment identifier in design file.")
    parser.add_argument("--fig", dest="ofig", action='store', required=True, help="Output figure name [pdf].")
    parser.add_argument("--fig2", dest="ofig2", action='store', required=True, help="Output figure name [html].")
    parser.add_argument("--debug", dest="debug", action='store_true', required=False, help="Add debugging log output.")
    args = parser.parse_args()
#     args = parser.parse_args(['--input', '/home/jfear/sandbox/secim/data/ST000015_AN000032_v2.txt',
#                               '--design', '/home/jfear/sandbox/secim/data/ST000015_design_v2.tsv',
#                               '--ID', 'Name',
#                               '--group', 'treatment',
#                               '--fig', '/home/jfear/sandbox/secim/data/test.pdf',
#                               '--fig2', '/home/jfear/sandbox/secim/data/test2.html',
#                               '--debug'])
    return(args)


def pltByTrt(dat, ax):
    """ Color Lines by Treatment """
    colors = pd.tools.plotting._get_standard_colors(len(dat.group))
    colLines = list()
    for i, grp in enumerate(dat.levels):
        samples = dat.design[dat.design[dat.group] == grp].index
        samp = dat.wide[samples]
        samp.plot(kind='kde', title='Distribution by Sample', ax=ax, c=colors[i])
        colLines.append(matplotlib.lines.Line2D([], [], c=colors[i], label=grp))

    return plt.legend(colLines, dat.levels, loc='upper left')


def main(args):
    # Import data
    logger.info(u'Importing data with following parameters: \n\tWide: {0}\n\tDesign: {1}\n\tUnique ID: {2}\n\tGroup Column: {3}'.format(args.fname, args.dname, args.uniqID, args.group))
    dat = wideToDesign(args.fname, args.dname, args.uniqID, args.group, clean_string=True)
    dat.wide.convert_objects(convert_numeric=True)

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    if hasattr(dat, 'group'):
        # If there is group information, go ahead and plot it.
        logger.info('Plotting sample distributions by group')
        legend1 = pltByTrt(dat, ax)
    else:
        logger.info('Plotting sample distributions')
        samp = dat.wide[dat.sampleIDs]
        samp.plot(kind='kde', title='Distribution by Sample', ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, ncol=5, loc='upper right', fontsize=10)

    try:
        ax.add_artist(legend1)
    except:
        pass

    plt.savefig(args.ofig, format='pdf')
    mpld3.save_html(fig, args.ofig2)


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
    else:
        sl.setLogger(logger)

    main(args)
