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


class FlagOutlier:
    """ Object for handling flags for outliers.

    Here we are iterating of pairwise combinations of samples and creating BA
    plots. If a sample falls outside of a 95% CI then want to flag these
    samples and outliers. We will have a flag for each pairwise combination.

    """
    def __init__(self, index):
        """
        Args:
            index (pd.Index): Index containing a list of compounds, will be
                used to create row index for flag table.

        Attributes:
            self.cnt (int): A counter for keep track of pairwise flag names.

            self.flag_outlier (pd.DataFrame): Data frame containing compound as
                row and flag name as columns.

            self.design (pd.DataFrame): Data frame relating flag name back to
                pairwise sample comparisons.

        """
        self.cnt = 0
        self.flag_outlier = pd.DataFrame(index=index)
        self.design = pd.DataFrame(index=[0], columns=('cbn1', 'cbn2'))

    def dropOutlier(self, data, bygroup=False, acrossAll=False):
        """ """
        if acrossAll:
            # Drop row if a flag is present in any sample.
            pass
        elif bygroup:
            # Set values for entire group to NaN if flag is present in any
            # sample within a group.
            pass
        else:
            # Set values to NaN if they were flagged.
            pass

        return

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
    """ Iterate over pairwise combinations and generate plots.

    This is a wrapper function to iterate over pairwise combinations and
    generate plots.

    Args:
        data (pd.DataFrame): Data frame containing data in wide format compound
            as rows and samples as columns.

        combos (list): List of tuples of pairwise combinations of samples.

        out (PdfPages): Handler for multi-page PDF that will contain all plots.

        flags (FlagOutlier): FlagOutlier object.

        group (str): Default is None if there are no groups. Otherwise it is
            the name of the current group. This value will be used in plot
            titles if present.

    Returns:
        out (PdfPages): A multi-page PDF containing scatter and BA plots.

        flags (FlagOutlier): Updated FlagOutlier object containing a table of
            outlier flags.

    """
    # Grab global counter
    for combo in combos:
        # Set up figure with 2 subplots
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


def runRegression(x, y):
    """ Run a linear regression.

    Args:
        x (pd.Series): Series of first sample, treated as independent variable.

        y (pd.Series): Series of second sample, treated as dependent variables.

    Returns:
        lower (pd.Series): Series of values for lower confidence interval.

        upper (pd.Series): Series of values for upper confidence interval.

        fitted (pd.Series): Series of fitted values.

    """
    model = sm.OLS(y, x)
    results = model.fit()
    fitted = results.fittedvalues
    prstd, lower, upper = wls_prediction_std(results)
    return lower, upper, fitted


def makeScatter(x, y, ax):
    """ Plot a scatter plot of x vs y.

    Args:
        x (pd.Series): Series of first sample, treated as independent variable.

        y (pd.Series): Series of second sample, treated as dependent variables.

        ax (matplotlib axis): Axis which to plot.

    Returns:
        ax (mapltolib axis): Axis with a scatter plot comparing x vs y with a
            regression line of fitted values (black) and the upper and lower
            confidence intervals (red).

    """
    # Get Names of the combo
    xname = x.name
    yname = y.name

    # Get Upper and Lower CI
    lower, upper, fitted = runRegression(x, y)

    # Plot
    ax.plot(x, y, 'o')
    ax.plot(x, lower, 'r--')
    ax.plot(x, fitted, 'k-')
    ax.plot(x, upper, 'r--')
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_title('Scatter plot')

    return ax


def makeBA(x, y, ax):
    """ Function to make BA Plot comparing x vs y.

    Args:
        x (pd.Series): Series of first sample, treated as independent variable.

        y (pd.Series): Series of second sample, treated as dependent variables.

        ax (matplotlib axis): Axis which to plot.

    Returns:
        ax (mapltolib axis): Axis with a Bland-Altman plot comparing x vs y with a
            horizontal line at y = 0 (black) and the upper and lower
            confidence intervals (red).

        mask (pd.Series BOOL): Series containing Boolean values with True
            indicating a value is more extreme than CI and should be an outlier and
            False indicating a value falls inside CI.

    """
    # Make BA plot
    diff = x - y
    mean = (x + y) / 2

    # Get Upper and Lower CI
    lower, upper, fitted = runRegression(mean, diff)
    mask = (diff >= upper) | (diff <= lower)

    # Plot
    ax.scatter(x=mean[~mask], y=diff[~mask])
    ax.scatter(x=mean[mask], y=diff[mask], color='r')

    ax.plot(mean, lower, 'r--')
    ax.axhline(0, color='k')
    ax.plot(mean, upper, 'r--')

    # Adjust axis
    ax.set_xlabel('Mean')
    ax.set_ylabel('Difference\n{0} - {1}'.format(x.name, y.name))
    ax.set_title('BA Plot')
    labels = ax.get_xmajorticklabels()
    plt.setp(labels, rotation=45)

    return ax, mask


def buildTitle(xname, yname, group):
    """ Build plot title.
    Args:
        xname (str): String containing the sampleID for x

        yname (str): String containing the sampleID for y

        gropu (str): String containing the group information. If no group then
            None.

    """
    if group:
        title = '{0}\n{1} vs {2}'.format(group, xname, yname)
    else:
        title = '{0} vs {1}'.format(xname, yname)

    return title


def main(args):
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

    # Drop outliers
    clean = flags.dropOutlier(wide, acrossAll=True)
    clean.to_table('/home/jfear/sandbox/secim/data/test.tsv')


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
    else:
        sl.setLogger(logger)

    main(args)
