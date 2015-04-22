#!/usr/bin/env python

# Built-in packages
from __future__ import division
import logging
import argparse
from argparse import RawDescriptionHelpFormatter
from itertools import combinations

# Add-on packages
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
    args = parser.parse_args(['--input', '/home/jfear/sandbox/secim/data/ST000015_log.tsv',
                              '--design', '/home/jfear/sandbox/secim/data/ST000015_design.tsv',
                              '--ID', 'Name',
                              '--group', 'treatment',
                              '--out', '/home/jfear/sandbox/secim/data/test.pdf',
                              '--debug'])
    return(args)


def runRegression(x, y):
    """ Run a simple linear regression """
    model = sm.OLS(y, x)
    results = model.fit()
    fitted = results.fittedvalues
    prstd, lower, upper = wls_prediction_std(results)
    return lower, upper, fitted


def makeScatter(x, y, ax):
    """ """

    # Get Names of the combo
    xname = x.name
    yname = y.name

    # Get Upper and Lower CI
    upper, lower, fitted = runRegression(x, y)

    # Plot
    ax.plot(x, y, 'o')
    ax.plot(x, fitted, 'k-')
    ax.plot(x, upper, 'r--')
    ax.plot(x, lower, 'r--')
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_title('Scatter plot')

    return ax


def makeBA(v1, v2, ax):
    """ Function to make BA Plot """

    # Make BA plot
    diff = v1 - v2
    mean = (v1 + v2) / 2

    # Get Upper and Lower CI
    lower, upper, fitted = runRegression(mean, diff)
    mask = (diff >= upper) | (diff <= lower)

    # Plot
    ax.scatter(x=mean[~mask], y=diff[~mask])
    ax.scatter(x=mean[mask], y=diff[mask], color='r')

    ax.axhline(0, color='k')
    ax.plot(mean, upper, 'r--')
    ax.plot(mean, lower, 'r--')

    # Adjust axis
    ax.set_xlabel('Mean')
    ax.set_ylabel('Difference\n{0} - {1}'.format(v1.name, v2.name))
    ax.set_title('BA Plot')
    labels = ax.get_xmajorticklabels()
    plt.setp(labels, rotation=45)

    return ax, mask


def buildTitle(xname, yname, group):
    """ Build plot title """
    if group:
        title = '{0}\n{1} vs {2}'.format(group, xname, yname)
    else:
        title = '{0} vs {1}'.format(xname, yname)

    return title


class FlagOutlier:
    """ """
    def __init__(self, index):
        """ """
        self.cnt = 0
        self.flag_outlier = pd.DataFrame(index=index)
        self.design = pd.DataFrame(index=[0], columns=('cbn1', 'cbn2'))

    def dropOutlier(self, data):
        """ """

    def updateOutlier(self, c1, c2, outlierMask):
        """ Update the flag_outlier object with a 0|1.

        Where 0 indicates a compound was not an outlier and 1 indicates that it
        was an outlier.

        Args:
            outlierMask (bool array-like): This is a boolean array-like object
                (i.e., pandas series) that is True for compounds that are outliers
                and False for compounds that are not outliers.

        Returns:
            self.flag_outlier: Updates self.flag_outlier with 0|1 flags.
            self.cnt: increments self.cnt by 1

        """

        # Update flag table
        self.flag_outlier.loc[outlierMask, 'flag_outlier{}'.format(self.cnt)] = 1
        self.flag_outlier.loc[~outlierMask, 'flag_outlier{}'.format(self.cnt)] = 0

        # Update design table
        self.design.loc[self.cnt, 'cbn1'] = c1
        self.design.loc[self.cnt, 'cbn2'] = c2
        self.cnt += 1


def iterateCombo(data, combos, out, flags, group=None):
    # Grab global counter

    for combo in combos:
        # Set up figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
        fig.subplots_adjust(wspace=0.4)

        # Scatter Plot
        ax1 = makeScatter(data[combo[0]], data[combo[1]], ax1)

        # BA plot
        ax2, outlier = makeBA(data[combo[0]], data[combo[1]], ax2)

        # Add plot title
        title = buildTitle(combo[0], combo[1], group)
        plt.suptitle(title, fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.90])

        # Output figure to pdf
        out.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Update Flags
        flags.updateOutlier(combo[0], combo[1], outlier)


def main(args):
    """ """
    # Import data
    dat = wideToDesign(args.fname, args.dname, args.uniqID, args.group)
    wide = dat.wide[dat.sampleIDs]
    flags = FlagOutlier(dat.wide.index)

    # Open a multiple page PDF for plots
    pp = PdfPages(args.oname)

    # If group is given, only do within group pairwise combinations
    if args.group:
        for i, val in dat.design.groupby(dat.group):
            combos = list(combinations(val.index, 2))
            iterateCombo(wide, combos, pp, flags, group=i)
    else:
        # Get all pairwise combinations for all samples
        combos = list(combinations(dat.sampleIDs, 2))
        iterateCombo(wide, combos, flags, pp)

    # Close PDF with plots
    pp.close()

    #
    print flags.flag_outlier.head(4)


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
    else:
        sl.setLogger(logger)

    main(args)
