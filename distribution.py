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
    description = """ Distribution Analysis: Plot sample distrubtions. """
    parser = argparse.ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)

    group1 = parser.add_argument_group(title='Standard input', description='Standard input for SECIM tools.')
    group1.add_argument("--input", dest="fname", action='store', required=True, help="Input dataset in wide format.")
    group1.add_argument("--design", dest="dname", action='store', required=True, help="Design file.")
    group1.add_argument("--ID", dest="uniqID", action='store', required=True, help="Name of the column with unique identifiers.")
    group1.add_argument("--fig", dest="ofig", action='store', required=True, help="Output figure name [pdf].")
    group1.add_argument("--fig2", dest="ofig2", action='store', required=True, help="Output figure name [html].")

    group2 = parser.add_argument_group(title='Optional input', description='Optional input for districution script.')
    group2.add_argument("--group", dest="group", action='store', required=False, help="Name of column in design file with Group/treatment information.")
    args = parser.parse_args()
    return(args)


def adjXCoord(ax):
    """ Adjust xlim on plots with no negative number.

    KDE of 0's can results in negative values, when data has no negative values
    then change xaxis to start at 0.

    """
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(0, xmax)


def pltByTrt(dat, ax):
    """ Color Lines by Group

    If group information is provided, then color distribution lines by group.

    """
    colors = pd.tools.plotting._get_standard_colors(len(dat.group))
    colLines = list()
    for i, grp in enumerate(dat.levels):
        samples = dat.design[dat.design[dat.group] == grp].index
        samp = dat.wide[samples]
        samp.plot(kind='kde', title='Distribution by Sample', ax=ax, c=colors[i])

        # Adjust coordinates if no negative values
        min = samp.values.min()
        if min >= 0:
            adjXCoord(ax)

        colLines.append(matplotlib.lines.Line2D([], [], c=colors[i], label=grp))

    return ax.legend(colLines, dat.levels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=10)


def pltBySample(dat, ax):
    """ Color Lines by Group

    If no group information then color distribution lines by sample.

    """
    samp = dat.wide[dat.sampleIDs]
    samp.plot(kind='kde', title='Distribution by Sample', ax=ax)

    # Adjust coordinates if no negative values
    min = samp.values.min()
    if min >= 0:
        adjXCoord(ax)


def pltBoxplot(dat, ax):
    """ Color Lines by Group

    If no group information then color distribution lines by sample.

    """
    samp = dat.wide[dat.sampleIDs]
    samp.boxplot(ax=ax, rot=45, return_type='dict')
    ax.set_title('Sample Distribution')


def main(args):
    # Import data
    logger.info(u'Importing data with following parameters: \n\tWide: {0}\n\tDesign: {1}\n\tUnique ID: {2}\n\tGroup Column: {3}'.format(args.fname, args.dname, args.uniqID, args.group))
    dat = wideToDesign(args.fname, args.dname, args.uniqID, args.group)
    dat.wide.convert_objects(convert_numeric=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20))
    plt.subplots_adjust(hspace=0.3)

    # If there is group information, color by group.
    if hasattr(dat, 'group'):
        logger.info('Plotting sample distributions by group')
        legend1 = pltByTrt(dat, ax1)
    else:
        logger.info('Plotting sample distributions')
        pltBySample(dat, ax1)

    # Create Legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, ncol=5, loc='upper right', fontsize=10)

    # Create second legend if there is group information
    if hasattr(dat, 'group'):
        ax1.add_artist(legend1)

    # Plot boxplot of samples
    pltBoxplot(dat, ax2)

    plt.savefig(args.ofig, format='pdf')
    mpld3.save_html(fig, args.ofig2, template_type='simple')


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    sl.setLogger(logger)

    main(args)
