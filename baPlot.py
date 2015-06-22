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
from interface import Flags
import logger as sl

# Globals
global DEBUG


def getOptions():
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

    args = parser.parse_args()

    # Check mutually inclusive options
    if (args.flagTable and not args.flagDesign) or (args.flagDesign and not args.flagTable):
        parser.error('--flag_table and --flag_design are both needed if you want to output raw pairwise flags.')

    return args


def summarizeFlags(dat, dfFlags, combos):
    """
    :type dat: wideToDesign
    :param dat:
    :type flags: pandas.DataFrame
    :param flags:
    :rtype: pandas.DataFrame
    :return:
    """

    # Create a data frame that is the same dimensions as wide. Each cell will be the sum of flags.
    flagSum = pd.DataFrame(index=dat.wide.index, columns=dat.wide.columns)
    flagSum.fillna(0)

    # Create a data frame to hold total possible for each cell.
    flagTotal = pd.DataFrame(index=dat.wide.index, columns=dat.wide.columns)
    flagTotal.fillna(0)

    # Iterate over sampleIDs and sum flags
    for sampleID in dat.sampleIDs:
        # Get list of flags that contain the current sampleID
        flagList = ['flag_{0}_{1}'.format(c[0], c[1]) for c in combos if sampleID in c]

        # Sum the flags in dfFlags for the current sampleID
        flagSum[:, sampleID] = dfFlags[flagList].sum(axis=1)

        # Get the totals of possible flags in dfFlags for the current sampleID
        flagTotal[:, sampleID] = dfFlags[flagList].count(axis=1)

    # Calculate the proportion of samples and features using the marginal sums.
    propSample = flagSum.sum(axis=0) / flagTotal.sum(axis=0)
    propFeature = flagSum.sum(axis=1) / flagTotal.sum(axis=1)

    return propSample, propFeature


def plotFlagDist(propSample, propFeature, pdf):
    """ Plot the distribution of flags.

    Sum outlier flags over samples and compounds and graph distribution.

    Arguments:
        :type summary: pandas.DataFrame
        :param summary: DataFrame of flags summarized to sample level.

        :param str out: Filename of pdf to save plots.

    Returns:
        :rtype: PdfPages
        :returns: Saves two bar plots to pdf.

    """
    # sort samples
    propSample.sort(ascending=False)

    # sort compounds
    propFeature.sort(ascending=False)

    # How many flags could I have
    nSample = propSample.shape[0]
    nFeature = propFeature.shape[0]

    # Make Plots
    ## Open pdf for plotting
    ppFlag = PdfPages(pdf)

    ## Plot samples
    propSample.head(30).plot(kind='bar', figsize=(10, 5))
    ppFlag.savefig(plt.gcf(), bbox_inches='tight')

    ## Plot compounds
    propFeature.head(30).plot(kind='bar', figsize=(10, 5))
    ppFlag.savefig(plt.gcf(), bbox_inches='tight')

    ## Close pdf
    ppFlag.close()


def buildTitle(dat, xName, yName):
    """ Build plot title.

    Arguments:
        :type dat: interface.wideToDesign
        :param dat: A wide to design object

        :param str xName: String containing the sampleID for x

        :param str yName: String containing the sampleID for y

    """
    if dat.design.loc[xName, :] == dat.design.loc[yName, :]:
        group = dat.design.loc[xName, :].values[0]
        title = '{0}\n{1} vs {2}'.format(group, xName, yName)
    else:
        title = '{0} vs {1}'.format(xName, yName)

    # Add on missing information if there are any missing values.
    try:
        if dat.missing == 1:
            title = title + '\n1 missing value'
        elif dat.missing > 0:
            title = title + '\n{} missing values'.format(dat.missing)
    except:
        pass

    return title


def runRegression(x, y):
    """ Run a linear regression.

    Arguments:
        :type x: pandas.Series
        :param x: Series of first sample, treated as independent variable.

        :type y: pandas.Series
        :param y: Series of second sample, treated as dependent variables.

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

    # Get 95% CI
    prstd, lower, upper = wls_prediction_std(results)

    return lower, upper, fitted, resid, influence


def makeBA(x, y, ax):
    """ Function to make BA Plot comparing x vs y.

    Arguments:
        :type x: pandas.Series
        :param x: Series of first sample, treated as independent variable.

        :type y: pandas.Series
        :param y: Series of second sample, treated as dependent variables.

        :type ax: matplotlib.axis
        :param ax: Axis which to plot.

    Returns:
        :rtype: pandas.Series
        :returns: A a series containing Boolean values with True
              indicating a value is more extreme than CI and should be an
              outlier and False indicating a value falls inside CI.

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

    # Create BA plot
    ax.scatter(x=mean[~mask], y=diff[~mask])
    ax.scatter(x=mean[mask], y=diff[mask], color='r', label='Outliers'.format(cutoff))
    ax.legend(loc='center left', bbox_to_anchor=(1, 1), fontsize=10)

    # Plot regression lines
    ax.plot(mean, lower, 'r:')
    ax.plot(mean, fitted, 'r')
    ax.axhline(0, color='k')
    ax.plot(mean, upper, 'r:')

    # Adjust axes
    ax.set_xlabel('Mean\n{0} & {1}'.format(x.name, y.name))
    ax.set_ylabel('Difference\n{0} - {1}'.format(x.name, y.name), fontsize=8)
    ax.set_title('Bland-Altman Plot')
    labels = ax.get_xmajorticklabels()
    plt.setp(labels, rotation=45)

    return mask


def makeScatter(x, y, ax):
    """ Plot a scatter plot of x vs y.

    Arguments:
        :type x: pandas.Series
        :param x: Series of first sample, treated as independent variable.

        :type y: pandas.Series
        :param y: Series of second sample, treated as dependent variables.

        :type ax: matplotlib.axis
        :param ax: Axis which to plot.

    """
    # Get Names of the combo
    xname = x.name
    yname = y.name

    #logger.debug('{0}, {1}'.format(x.name, y.name))
    # Get Upper and Lower CI
    lower, upper, fitted, resid, infl = runRegression(x, y)

    # Plot scatter
    ax.scatter(x, y)

    # Plot regression lines
    ax.plot(x, lower, 'r:')
    ax.plot(x, fitted, 'r-')
    ax.plot(x, upper, 'r:')

    # Adjust plot
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_title('Scatter plot')


def iterateCombo(dat, combo, pdf):
    """ Iterate over pairwise combinations and generate plots.

    This is a wrapper function to iterate over pairwise combinations and
    generate plots.

    Arguments:
        :type data: pandas.DataFrame
        :param data: Data frame containing data in wide format compound
            as rows and samples as columns.

        :param list combos: List of tuples of pairwise combinations of samples.

        :type out: PdfPages
        :param out: Handler for multi-page PDF that will contain all plots.

        :type flags: FlagOutlier
        :param flags: FlagOutlier object.

        :param str group: Default is None if there are no groups. Otherwise it is
            the name of the current group. This value will be used in plot
            titles if present.

    Updates:
        :type out: PdfPages
        :param out: Handler for multi-page PDF that will contain all plots.

        :type flags: FlagOutlier
        :param flags: FlagOutlier object.

    """

    # Current combinations
    c1 = combo[0]
    c2 = combo[1]

    # Set up figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7), dpi=300)
    fig.subplots_adjust(wspace=0.4)

    # Scatter Plot of c1 vs c2
    makeScatter(dat.wide.loc[:, c1], dat.wide.loc[:, c2], ax1)

    # BA plot of c1 vs c2
    outlier = makeBA(dat.wide.loc[:, c1], dat.wide.loc[:, c2], ax2)

    # Add plot title
    title = buildTitle(dat, c1, c2)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])

    # Output figure to pdf
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # Create flags
    flag = Flags(dat.wide.index, 'flag_{0}_{1}'.format(c1, c2))
    flag.update(outlier)

    return flag


def main(args):
    # Import data
    logger.info('Importing Data')
    dat = wideToDesign(args.fname, args.dname, args.uniqID, args.group)

    # Get a list of samples to process, if processOnly is specified only analyze this group.
    if args.processOnly:
        dat.design = dat.design[dat.design[args.group].isin(args.processOnly)]
        toProcess = dat.design.index
        dat.sampleIDs = toProcess.tolist()
    else:
        # Process everything
        toProcess = dat.sampleIDs

    # Create dataframe with sampleIDs that are to be analyzed.
    #: :type wide: pandas.DataFrame
    dat.keep_sample(toProcess)

    # TODO: Add better missing data handling.
    # Drop rows with missing data
    rowNum = dat.wide.shape[0]
    dat.wide.dropna(inplace=True)
    dat.missing = rowNum - dat.wide.shape[0]
    logger.warn(""" There were {} rows with missing data, please
                    make sure you have run missing data script
                    before running baPlot. """.format(dat.missing))

    # Get list of pairwise combinations. If group is specified, only do within group combinations.
    combos = list()
    if args.group:
        # If group is given, only do within group pairwise combinations
        logger.info('Only doing within group, pairwise comparisons.')
        for groupName, dfGroup in dat.design.groupby(dat.group):
            combos.append(list(combinations(dfGroup.index, 2)))
    else:
        logger.info('Doing all pairwise comparisons. This could take a while!')
        # Get all pairwise combinations for all samples
        combos.append(list(combinations(dat.sampleIDs, 2)))

    # Open a multiple page PDF for plots
    ppBA = PdfPages(args.baName)

    # Loop over combinations and generate plots and return a list of flags.
    # Also generates several figures that are output to the multi-page pdf.
    flags = map(lambda combo: iterateCombo(dat, combo, ppBA), combos)

    # Close PDF with plots
    ppBA.close()

    # Merge flags
    merged = Flags.merge(flags)

    # Summarize flags
    logger.info('Summarizing outlier flags.')
    propSample, propFeature = summarizeFlags(dat, merged, combos)
    plotFlagDist(propSample, propFeature, args.distName)

    # Create metabolite level flags
    flag_metabolite = Flags(dat.wide.index, 'flag_feature_BA_outlier')
    flag_metabolite.update((propFeature >= .5))
    flag_metabolite.to_csv('/home/jfear/tmp/flag_met.csv')

    # Create sample level flags
    flag_sample = Flags(dat.wide.index, 'flag_sample_BA_outlier')
    flag_sample.update((propSample >= .5))
    flag_sample.to_csv('/home/jfear/tmp/flag_met.csv')

    # Output Raw Flags
    if args.flagTable and args.flagDesign:
        logger.info('Outputting raw outlier flags.')
        merged.to_csv(args.flagTable, sep='\t')


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    # Expose cutoff as global
    global cutoff
    cutoff = args.cutoff

    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
        DEBUG = True
    else:
        DEBUG = False
        sl.setLogger(logger)

    main(args)
