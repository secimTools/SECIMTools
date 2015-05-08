#!/usr/bin/env python

# Built-in packages
from __future__ import division
import logging
import argparse
from argparse import RawDescriptionHelpFormatter
from itertools import combinations

# Add-on packages
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# Local Packages
from interface import wideToDesign
import logger as sl

# Globals
global DEBUG
DEBUG = False


def getOptions(myopts=None):
    """ Function to pull in arguments """
    description = """ The Bland-Altman plot (BA-plot) is commonly used to look
    at concordance of data between samples. It is especially useful for looking
    at variability between replicates. This script will generate BA-plots for
    all pairwise combinations of samples, or if group information is provided
    it will only report pairwise combinations within the group.

    A linear regression is also performed on the BA-plots to identify samples
    whose residuals are beyond a cutoff. For each compound (row) in the
    dataset, a sample is flagged as an outlier if the Pearson normalized
    residuals are greater than a cutoff (--filter_cutoff).

    These raw flags can be output by specifying the --flag_table and
    --flag_design options. flag_table is a table with compounds as rows and
    flag_outlier## as columns, where the ## is incremented for each pairwise
    comparison between samples. flag_design can then be used to relate sample
    to its corresponding flags.

    The script always outputs a summary of the flags, where the flags for each
    sample is summed across all comparisons.

    Two sets of plots are output: (1) Bland-Altman plots for pairwise
    comparisons are saved to a pdf specified by (--ba). (2) Bar graphs of
    summarized flags are saved by (--flag_summary).

    """
    parser = argparse.ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)
    group1 = parser.add_argument_group(title='Standard input', description='Standard input for SECIM tools.')
    group1.add_argument("--input", dest="fname", action='store', required=True, help="Input dataset in wide format.")
    group1.add_argument("--design", dest="dname", action='store', required=True, help="Design file.")
    group1.add_argument("--ID", dest="uniqID", action='store', required=True, help="Name of the column with unique identifiers.")
    group1.add_argument("--group", dest="group", action='store', default=False, required=False, help="Group/treatment identifier in design file [Optional].")

    group2 = parser.add_argument_group(title='Required input', description='Additional required input for this tool.')
    group2.add_argument("--ba", dest="baName", action='store', required=True, help="Name of the output PDF for Bland-Altman plots.")
    group2.add_argument("--flag_dist", dest="distName", action='store', required=True, help="Name of the output PDF for plots.")
    group2.add_argument("--flag_summary", dest="flagSummary", action='store', required=True, help="Output table summarizing flags.")

    group3 = parser.add_argument_group(title='Optional Settings')
    group3.add_argument("--process_only", dest="processOnly", action='store', nargs='+', default=False, required=False, help="Only process the given groups (list groups separated by spaces) [Optional].")
    group3.add_argument("--filter_cutoff", dest="cutoff", action='store', default=3, type=int, required=False, help="Cutoff value for flagging outliers [default=3].")
    group3.add_argument("--flag_table", dest="flagTable", action='store', required=False, help="Output pairwise flags in Tabular format [Optional].")
    group3.add_argument("--flag_design", dest="flagDesign", action='store', required=False, help="Output table relating pairwise flags back to sample ID [Optional].")

    group4 = parser.add_argument_group(title='Development Settings')
    group4.add_argument("--debug", dest="debug", action='store_true', required=False, help="Add debugging log output.")

    if not myopts:
        args = parser.parse_args()
    else:
        args = parser.parse_args(myopts)

#     args = parser.parse_args(['--input', '/home/jfear/sandbox/secim/data/ST000015_log.tsv',
#                               '--design', '/home/jfear/sandbox/secim/data/ST000015_design.tsv',
#                               '--ID', 'Name',
#                               '--group', 'treatment',
#                               '--ba', '/home/jfear/sandbox/secim/data/test_ba.pdf',
#                               '--flag_dist', '/home/jfear/sandbox/secim/data/test_dist.pdf',
#                               '--flag_summary', '/home/jfear/sandbox/secim/data/test_flag_summary.tsv',
#                               '--process_only', '02_uM_palmita',
#                               '--debug'])

#     args = parser.parse_args(['--input', '/home/jfear/sandbox/secim/data/BR1_FTA_43_peak_intensity_data_log.tsv',
#                               '--design', '/home/jfear/sandbox/secim/data/BR1_FTA_43_design.tsv',
#                               '--ID', 'ppm',
#                               '--group', 'groupID',
#                               '--ba', '/home/jfear/sandbox/secim/data/test_ba.pdf',
#                               '--flag_dist', '/home/jfear/sandbox/secim/data/test_dist.pdf',
#                               '--flag_summary', '/home/jfear/sandbox/secim/data/test_flag_summary.tsv',
#                               '--debug'])

    # Check mutually inclusive options
    if (args.flagTable and not args.flagDesign) or (args.flagDesign and not args.flagTable):
        parser.error('--flag_table and --flag_design are both needed if you want to output raw pairwise flags.')

    return(args)


class FlagOutlier:
    """ Object for handling flags for outliers.

    Iterate over pairwise combinations of samples and create BA
    plots. If a sample falls outside of a 95% CI then flag the sample as an
    outlier. This class is used to store flag information an provides a few
    basic methods for interacting with flags.

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
        self.flag_outlier = pd.DataFrame(index=index, dtype='int64')
        self.design = pd.DataFrame(index=[0], columns=('cbn1', 'cbn2'))

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

        # Convert outlierMask to dataframe
        outlierMask.name = 'flag_outlier{}'.format(self.cnt)
        oDF = pd.DataFrame(outlierMask, dtype='int64')

        # Merge to flag table
        self.flag_outlier = self.flag_outlier.join(oDF, how='outer')
        self.flag_outlier.fillna(0)

        # Update design table
        self.design.loc[self.cnt, 'cbn1'] = c1
        self.design.loc[self.cnt, 'cbn2'] = c2
        self.cnt += 1

    def summarizeSampleFlags(self, data):
        """ Summarize flag_outlier to the sample level for easy qc.

        Args:
            data (pd.DataFrame): Wide formatted dataframe that is used to get
                row and column labels for flag summary.

        Returns:
            summary (pd.DataFrame): Dataframe where values are the sum of flags
                for pairwise comparison.

        """

        # Create a new dataframe to hold summarized flags
        summary = pd.DataFrame(index=data.index)

        ## Iterate over samples
        for sample in data.columns:
            # Which pairwise flags correspond to the current sample
            pflag = (self.design['cbn1'] == sample) | (self.design['cbn2'] == sample)

            # Pull flag_outlier columns that correspond to sample
            cols = self.flag_outlier.columns[pflag]
            currFlags = self.flag_outlier[cols]

            # Sum across columns
            sums = currFlags.apply(np.sum, axis=1)
            sums.name = sample

            # Update summary dataset
            summary = summary.join(sums)

        summary.fillna(0, inplace=True)

        return summary


def iterateCombo(data, combos, out, flags, cutoff, group=None):
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
    # How many rows are there in total
    rows = data.shape[0]

    # Grab global counter
    for combo in combos:
        subset = data.loc[:, [combo[0], combo[1]]]

        # Drop missing value
        subset.dropna(inplace=True)
        missing = rows - subset.shape[0]

        # Set up figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7), dpi=300)
        fig.subplots_adjust(wspace=0.4)

        # Scatter Plot
        ax1 = makeScatter(subset.iloc[:, 0], subset.iloc[:, 1], ax1)

        # BA plot
        ax2, outlier = makeBA(subset.iloc[:, 0], subset.iloc[:, 1], ax2, cutoff)

        # Add plot title
        title = buildTitle(combo[0], combo[1], group, missing)
        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.90])

        # Output figure to pdf
        out.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Update Flags
        flags.updateOutlier(combo[0], combo[1], outlier)


def plotLeverageDensity(infl):
    """ For debugging we want to look at density of leverage """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    infl['cooks_pval'].plot(kind='kde', ax=ax)
    plt.savefig('/home/jfear/tmp/density.pdf')
    plt.close(fig)


def runRegression(x, y):
    """ Run a linear regression.

    Args:
        x (pd.Series): Series of first sample, treated as independent variable.

        y (pd.Series): Series of second sample, treated as dependent variables.

    Returns:
        lower (pd.Series): Series of values for lower confidence interval.

        upper (pd.Series): Series of values for upper confidence interval.

        fitted (pd.Series): Series of fitted values.

        resid (pd.DataFrame): DataFrame containing residuals and Pearson
            normalized residuals to have unit variance.

    """
    # Fit linear regression
    model = sm.OLS(y, x)
    results = model.fit()
    fitted = results.fittedvalues
    infl = results.get_influence()
    CD = infl.cooks_distance
    (dffits, thresh) = infl.dffits
    DF = dffits > thresh

    influence = pd.DataFrame({'cooksD': CD[0], 'cooks_pval': CD[1], 'dffits': DF}, index=fitted.index)

    # Pull Residuals
    presid = pd.Series(results.resid_pearson, index=results.resid.index)
    resid = pd.concat([results.resid, presid], axis=1)
    resid.columns = pd.Index(['resid', 'resid_pearson'])

    # Plot density of stats if debug
    if DEBUG:
        plotLeverageDensity(influence)

    # Get 95% CI
    prstd, lower, upper = wls_prediction_std(results)

    return lower, upper, fitted, resid, influence


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

    #logger.debug('{0}, {1}'.format(x.name, y.name))
    # Get Upper and Lower CI
    lower, upper, fitted, resid, infl = runRegression(x, y)

    # Plot
    ax.plot(x, y, 'o')
    ax.plot(x, lower, 'r:')
    ax.plot(x, fitted, 'r-')
    ax.plot(x, upper, 'r:')
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_title('Scatter plot')

    return ax


def makeBA(x, y, ax, cutoff):
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
    lower, upper, fitted, resid, infl = runRegression(mean, diff)
    mask1 = abs(resid['resid_pearson']) > cutoff
    mask2 = infl['cooks_pval'] <= 0.5
    mask3 = infl['dffits']
    mask = mask1 | mask2 | mask3
    #maskC12 = mask1 & mask2
    #maskC13 = mask1 & mask3
    #maskC23 = mask2 & mask3
    #maskC123 = mask1 & mask2 & mask3

    # Plot
    ax.scatter(x=mean[~mask], y=diff[~mask])
    ax.scatter(x=mean[mask], y=diff[mask], color='r', label='Outliers'.format(cutoff))
    #ax.scatter(x=mean[mask1], y=diff[mask1], color='r', label='Outliers [abs(residual) > {}]'.format(cutoff))
    #ax.scatter(x=mean[mask2], y=diff[mask2], color='g', label='Leverage [cooksD p-val > 0.5)')
    #ax.scatter(x=mean[mask3], y=diff[mask3], color='k', label='Dffits')
    #ax.scatter(x=mean[maskC12], y=diff[maskC12], color='c', label='Both Outlier and Cooks')
    #ax.scatter(x=mean[maskC13], y=diff[maskC13], color='m', label='Both Outlier and Dffits')
    #ax.scatter(x=mean[maskC23], y=diff[maskC23], color='y', label='Both Cooks and Dffits')
    #ax.scatter(x=mean[maskC123], y=diff[maskC123], color='r', marker='s', s=50, label='Outlier, Cooks, and Dffits')
    ax.legend(loc='center left', bbox_to_anchor=(1, 1), fontsize=10)

    #ax.plot(mean, lower, 'r:')
    ax.plot(mean, fitted, 'r')
    ax.axhline(0, color='k')
    #ax.plot(mean, upper, 'r:')

    # Adjust axis
    ax.set_xlabel('Mean\n{0} & {1}'.format(x.name, y.name))
    ax.set_ylabel('Difference\n{0} - {1}'.format(x.name, y.name), fontsize=8)
    ax.set_title('Bland-Altman Plot')
    labels = ax.get_xmajorticklabels()
    plt.setp(labels, rotation=45)

    return ax, mask


def buildTitle(xname, yname, group, missing):
    """ Build plot title.
    Args:
        xname (str): String containing the sampleID for x

        yname (str): String containing the sampleID for y

        group (str): String containing the group information. If no group then
            None.

    """
    if group:
        title = '{0}\n{1} vs {2}'.format(group, xname, yname)
    else:
        title = '{0} vs {1}'.format(xname, yname)

    # Add on missing information if there are any missing values.
    if missing == 1:
        title = title + '\n1 missing value'
    elif missing > 0:
        title = title + '\n{} missing values'.format(missing)

    return title


def plotFlagDist(summary, out):
    """ Plot the distribution of flags.

    Sum outlier flags over samples and compounds and graph distribution.

    Args:
        summary (pd.DataFrame): DataFrame of flags summarized to sample level.

        out (str): Filename of pdf to save plots.
    Returns:
        Saves two bar plots to pdf.

    """
    # Sum flags across samples
    col_sum = summary.sum(axis=0)
    col_sum.sort(ascending=False)

    # Sum flags across compounds
    row_sum = summary.sum(axis=1)
    row_sum.sort(ascending=False)

    # How many flags could I have
    row_max, col_max = summary.shape

    # Make Plots
    ## Open pdf for plotting
    ppFlag = PdfPages(out)

    ## Plot samples
    if np.any(col_sum > 0):
        col_sum[col_sum > 0].plot(kind='bar', figsize=(10, 5))
        ppFlag.savefig(plt.gcf(), bbox_inches='tight')

    ## Plot compounds
    if np.any(row_sum > 0):
        row_sum[row_sum > 0].head(30).plot(kind='bar', figsize=(10, 5))
        ppFlag.savefig(plt.gcf(), bbox_inches='tight')

    ## Close pdf
    ppFlag.close()


def convertToInt(x):
    """ Convert to integer before export """
    try:
        return x.astype(int)
    except:
        return x


def main(args):
    print args
    # Import data
    logger.info('Importing Data')
    dat = wideToDesign(args.fname, args.dname, args.uniqID, args.group)

    # Keep only columns that are specified by processOnly group, or sampleIDs
    if args.processOnly:
        dat.design = dat.design[dat.design[args.group].isin(args.processOnly)]
        toProcess = dat.design.index
        dat.sampleIDs = toProcess.tolist()
    else:
        # Process everything
        toProcess = dat.sampleIDs

    wide = dat.wide[toProcess]

    # Create a FlagOutlier object to store all flags
    flags = FlagOutlier(dat.wide.index)

    # Open a multiple page PDF for plots
    ppBA = PdfPages(args.baName)

    # If group is given, only do within group pairwise combinations
    if args.group:
        logger.info('Only doing within group, pairwise comparisons.')
        for i, val in dat.design.groupby(dat.group):
            combos = list(combinations(val.index, 2))
            iterateCombo(wide, combos, ppBA, flags, args.cutoff, group=i)
    else:
        logger.info('Doing all pairwise comparisons. This could take a while!')
        # Get all pairwise combinations for all samples
        combos = list(combinations(dat.sampleIDs, 2))
        iterateCombo(wide, combos, ppBA, flags, args.cutoff, group=None)

    # Close PDF with plots
    ppBA.close()

    # Summarize flags
    logger.info('Summarizing outlier flags.')
    summary = flags.summarizeSampleFlags(wide)
    plotFlagDist(summary, args.distName)

    summary2 = summary.astype('int64')
    summary2.to_csv(args.flagSummary, sep='\t')

    # Output Raw Flags
    if args.flagTable and args.flagDesign:
        logger.info('Outputing raw outlier flags.')
        flags.flag_outlier.fillna(0, inplace=True)
        flag_outlier = flags.flag_outlier.astype('int64')
        flag_outlier.to_csv(args.flagTable, sep='\t')
        flags.design.to_csv(args.flagDesign, sep='\t')


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
        DEBUG = True
    else:
        sl.setLogger(logger)

    main(args)
