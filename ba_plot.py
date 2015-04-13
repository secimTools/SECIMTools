#!/usr/bin/env python

# Built-in packages
from __future__ import division
import logging
import argparse
from argparse import RawDescriptionHelpFormatter
from itertools import combinations

# Add-on packages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

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
    parser.add_argument("--group", dest="group", action='store', default=False, required=False, help="Group/treatment identifier in design file.")
    parser.add_argument("--out", dest="oname", action='store', required=True, help="Name of the output PDF")
    parser.add_argument("--debug", dest="debug", action='store_true', required=False, help="Add debugging log output.")
#     args = parser.parse_args()
    args = parser.parse_args(['--input', '/home/jfear/sandbox/secim/data/ST000015.tsv',
                              '--design', '/home/jfear/sandbox/secim/data/ST000015_design.tsv',
                              '--ID', 'Name',
                              '--group', 'treatment',
                              '--out', '/home/jfear/sandbox/secim/data/test.pdf',
                              '--debug'])
    return(args)


def makeBA(v1, v2, ax):
    """ Function to make BA Plot """
    # Make BA plot
    diff = v1 - v2
    mean = (v1 + v2) / 2
    ax.axhline(0, color='r')
    ax.scatter(x=mean, y=diff)

    # Adjust axis
    ax.set_xlabel('Mean')
    ax.set_ylabel('Difference')
    labels = ax.get_xmajorticklabels()
    plt.setp(labels, rotation=45)

    return ax


def buildTitle(combo, group, design):
    """ Build plot title """
    if group:
        title = u'{0}\n{1} vs {2}'.format(design.ix[combo[0], group], combo[0], combo[1])
    else:
        title = '{} vs {}'.format(*combo)

    return title


def main(args):
    """ """
    # Import data
    dat = wideToDesign(args.fname, args.dname, args.uniqID, args.group)
    wide = dat.wide[dat.sampleIDs]

    # If group is given, only do within group pairwise combinations
    if args.group:
        combos = list()
        for i, val in dat.design.groupby(dat.group):
            combos.extend(list(combinations(val.index, 2)))
    else:
        # Get all pairwise combinations for all samples
        combos = combinations(dat.sampleIDs, 2)

    # Open output pdf
    pp = PdfPages(args.oname)

    for combo in combos:
        # Set up figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6), dpi=300)
        fig.subplots_adjust(wspace=0.4)

        # Build regression plot
        sns.lmplot(combo[0], combo[1], wide, ax=ax1)
        labels = ax1.get_xmajorticklabels()
        plt.setp(labels, rotation=45)

        # Build BA plot
        ax2 = makeBA(wide[combo[0]], wide[combo[1]], ax2)

        # Add plot title
        title = buildTitle(combo, args.group, dat.design)
        ax2.set_title(title, fontsize=12)

        # Regression by Hand
        x = wide[combo[0]]
        y = wide[combo[1]]

        model = sm.OLS(y, x)
        results = model.fit()
        prstd, iv_l, iv_u = wls_prediction_std(results)

        ax3.plot(x, y, 'o')
        ax3.plot(x, results.fittedvalues, 'k-')
        ax3.plot(x, iv_u, 'r--')
        ax3.plot(x, iv_l, 'r--')

        # Output figure to pdf
        pp.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        break

    pp.close()


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
    else:
        sl.setLogger(logger)

    main(args)
