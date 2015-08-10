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


def getOptions():
    """ Function to pull in arguments """

    description = """ The Bland-Altman plot (BA-plot) is commonly used to look
    at concordance of samples. It is useful for looking at variability between
    replicates. This script generates BA-plots for all pairwise combinations of
    samples, or if group information is provided it will only report pairwise
    combinations within the group.

    A linear regression is also performed on the BA-plots to identify samples
    whose residuals are beyond a cutoff. For each feature (row) in the dataset,
    a sample is flagged as an outlier if the Pearson normalized residuals are
    greater than a cutoff (--filter_cutoff). Or if the leverage statistics
    (DFFITS and Cook's D) flag the feature as a leverage point.

    The script outputs a separate set of the flags for samples and features.

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
    group2.add_argument("--flag_sample", dest="flagSample", action='store', required=True, help="Name of the output TSV for sample flags.")
    group2.add_argument("--flag_feature", dest="flagFeature", action='store', required=True, help="Name of the output TSV for feature flags.")

    group3 = parser.add_argument_group(title='Optional Settings')
    group3.add_argument("--process_only", dest="processOnly", action='store', nargs='+', default=False, required=False, help="Only process the given groups (list groups separated by spaces) [Optional].")
    group3.add_argument("--resid_cutoff", dest="residCutoff", action='store', default=3, type=int, required=False, help="Cutoff value for flagging outliers [default=3].")

    group3.add_argument("--sample_flag_cutoff", dest="sampleCutoff", action='store', default=.20, type=float, required=False, help="Proportion cutoff value when flagging samples [default=0.20].")
    group3.add_argument("--feature_flag_cutoff", dest="featureCutoff", action='store', default=.05, type=float, required=False, help="Proportion cutoff value when flagging features [default=0.05].")

    group4 = parser.add_argument_group(title='Development Settings')
    group4.add_argument("--debug", dest="debug", action='store_true', required=False, help="Add debugging log output.")

    args = parser.parse_args()

    if (args.sampleCutoff > 1) | (args.sampleCutoff < 0):
        parser.error('sample_flag_cutoff must be a number between 0 and 1')

    if (args.featureCutoff > 1) | (args.featureCutoff < 0):
        parser.error('feature_flag_cutoff must be a number between 0 and 1')

    return args


def summarizeFlags(dat, dfFlags, combos):
    """ Given a set of flags calculate the proportion of times a feature is flagged.

    :Arguments:
        :type dat: interface.wideToDesign
        :param dat: A wideToDesign object that will be used to get sample and
            feature information.

        :type flags: pandas.DataFrame
        :param flags: A Dataframe of flags for all pairwise comparisons.

    :Returns:
        :rtype: tuple of pandas.DataFrame
        :return: Two DataFrames, the first has the proportion of samples that were
            flagged. The second has the proportion of features flagged.

    """
    # Create a data frame that is the same dimensions as wide. Where each cell
    # will be the sum of flags.
    flagSum = pd.DataFrame(index=dfFlags.index, columns=dat.wide.columns)
    flagSum.fillna(0, inplace=True)

    # Create a data frame that is the same dimensions as wide. Where each cell
    # is the total number of comparisons.
    flagTotal = pd.DataFrame(index=dfFlags.index, columns=dat.wide.columns)
    flagTotal.fillna(0, inplace=True)

    for sampleID in dat.sampleIDs:
        # Get list of flags that contain the current sampleID
        flagList = ['flag_{0}_{1}'.format(c[0], c[1]) for c in combos if sampleID in c]

        # Sum the flags in dfFlags for the current sampleID
        flagSum.ix[:, sampleID] = dfFlags[flagList].sum(axis=1).values

        # Get the totals of possible flags in dfFlags for the current sampleID
        flagTotal.ix[:, sampleID] = dfFlags[flagList].count(axis=1).values

    # Calculate the proportion of samples and features using the marginal sums.
    propSample = flagSum.sum(axis=0) / flagTotal.sum(axis=0)
    propFeature = flagSum.sum(axis=1) / flagTotal.sum(axis=1)

    return propSample, propFeature


def plotFlagDist(propSample, propFeature, pdf):
    """ Plot the distribution of proportion of samples and features that were outliers.

    :Arguments:
        :type propSample: pandas.DataFrame
        :param propSample: Data frame of the proportion of samples flagged as
            an outlier.

        :type propFeature: pandas.DataFrame
        :param propFeature: Data frame of the proportion of features flagged as
            an outlier.

        :param str pdf: Filename of pdf to save plots.

    :Returns:
        :rtype: matplotlib.backends.backend_pdf.PdfPages
        :returns: Saves two bar plots to pdf.

    """
    # sort samples
    propSample.sort(ascending=False)

    # sort compounds
    propFeature.sort(ascending=False)

    # Make Plots
    ## Open pdf for plotting
    ppFlag = PdfPages(pdf)

    ## Plot samples
    ax = propSample.head(30).plot(kind='bar', figsize=(10, 5), rot=90)
    ax.set_ylabel('Proportion of features that were outliers.')
    ax.set_xlabel('Sample ID')
    ppFlag.savefig(plt.gcf(), bbox_inches='tight')

    ## Plot features
    ax = propFeature.head(30).plot(kind='bar', figsize=(10, 5), rot=90)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Proportion of samples that a feature was an outlier.')
    ax.set_xlabel('Feature ID')
    ppFlag.savefig(plt.gcf(), bbox_inches='tight')

    ## Close pdf
    ppFlag.close()


def buildTitle(dat, xName, yName):
    """ Build plot title.

    :Arguments:
        :type dat: interface.wideToDesign
        :param dat: A wide to design object

        :param str xName: String containing the sampleID for x

        :param str yName: String containing the sampleID for y

    :Returns:
        :rtype: str
        :returns: A string containing the plot title.
    """
    # If groups are the same, add group information to the plot.
    if dat.group and dat.design.loc[xName, dat.group] == dat.design.loc[yName, dat.group]:
        group = dat.design.loc[xName, dat.group]
        title = '{0}\n{1} vs {2}'.format(group, xName, yName)
    else:
        title = '{0} vs {1}'.format(xName, yName)

    # If there are missing values add the count to the title.
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

    :Arguments:
        :type x: pandas.Series
        :param x: Series of first sample, treated as independent variable.

        :type y: pandas.Series
        :param y: Series of second sample, treated as dependent variables.

    :Returns:
        :rtype: tuple of pandas.Series and pandas.DataFrame
        :returns: A tuple of Series and data frames:
            * lower (pd.Series): Series of values for lower confidence interval.
            * upper (pd.Series): Series of values for upper confidence interval.
            * fitted (pd.Series): Series of fitted values.
            * resid (pd.DataFrame): DataFrame containing residuals and Pearson
              normalized residuals to have unit variance.

    """
    # Fit linear regression
    # Drop missing values for the regression
    model = sm.OLS(y, x, missing='drop')
    results = model.fit()

    # Get fit and influence stats
    fitted = results.fittedvalues
    infl = results.get_influence()
    CD = infl.cooks_distance
    (dffits, thresh) = infl.dffits
    DF = abs(dffits) > thresh

    influence = pd.DataFrame({'cooksD': CD[0], 'cooks_pval': CD[1], 'dffits': DF}, index=fitted.index)

    # Get Residuals
    presid = pd.Series(results.resid_pearson, index=results.resid.index)
    resid = pd.concat([results.resid, presid], axis=1)
    resid.columns = pd.Index(['resid', 'resid_pearson'])

    # Get 95% CI
    prstd, lower, upper = wls_prediction_std(results)

    return lower, upper, fitted, resid, influence


def makeBA(x, y, ax):
    """ Function to make BA Plot comparing x vs y.

    :Arguments:
        :type x: pandas.Series
        :param x: Series of first sample, treated as independent variable.

        :type y: pandas.Series
        :param y: Series of second sample, treated as dependent variables.

        :type ax: matplotlib.axis
        :param ax: Axis which to plot.

    :Returns:
        :rtype: pandas.Series
        :returns: A Series containing Boolean values with True
              indicating a value is more extreme than CI and False indicating a
              value falls inside CI.

    """
    # Make BA plot
    diff = x - y
    mean = (x + y) / 2

    # Drop missing for current comparison
    diff.dropna(inplace=True)
    mean.dropna(inplace=True)

    # Get Upper and Lower CI from regression
    lower, upper, fitted, resid, infl = runRegression(mean, diff)
    mask1 = abs(resid['resid_pearson']) > cutoff
    mask2 = infl['cooks_pval'] <= 0.5
    mask3 = infl['dffits']
    mask = mask1 | mask2 | mask3

    # Create BA plot
    ax.scatter(x=mean[~mask], y=diff[~mask])
    ax.scatter(x=mean[mask], y=diff[mask], color='r', label='Leverage Points'.format(cutoff))
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

    :Arguments:
        :type x: pandas.Series
        :param x: Series of first sample, treated as independent variable.

        :type y: pandas.Series
        :param y: Series of second sample, treated as dependent variables.

        :type ax: matplotlib.axis
        :param ax: Axis which to plot.

    :Returns:
        :rtype: matplotlib.axis
        :returns: A matplotlib axis with a scatter plot.

    """
    # Get Names of the combo
    xname = x.name
    yname = y.name

    #logger.debug('{0}, {1}'.format(x.name, y.name))
    # Get Upper and Lower CI from regression
    lower, upper, fitted, resid, infl = runRegression(x, y)

    # Plot scatter
    ax.scatter(x, y)

    # Plot regression lines
    # If there are missing data, x and the result vectors won't have the same
    # dimensions. First filter x by the index of the fitted values then plot.
    x2 = x.loc[fitted.index]
    ax.plot(x2, lower, 'r:')
    ax.plot(x2, fitted, 'r-')
    ax.plot(x2, upper, 'r:')

    # Adjust plot
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_title('Scatter plot')


def iterateCombo(dat, combo, pdf):
    """ A function to iterate generate all plots and flags.

    :Arguments:
        :type dat: interface.wideToDesign
        :param dat: A wideToDesign object containing wide and design information.

        :param tuple combo: A tuple of pairwise combination for current sample.

        :type pdf: matplotlib.backends.backend_pdf.PdfPages
        :param pdf: Handler for multi-page PDF that will contain all plots.

    :Updates:
        :type pdf: matplotlib.backends.backend_pdf.PdfPages
        :param pdf: Handler for multi-page PDF that will contain all plots.

    :Returns:
        :rtype flag: interface.Flags
        :param flag: A Flags object with outlier flags.

    """

    # Current combination
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
    flag = Flags(index=dat.wide.index)
    flag.addColumn(column='flag_{0}_{1}'.format(c1, c2), mask=outlier)

    return flag.df_flags


def main(args):
    # Import data
    logger.info('Importing Data')
    dat = wideToDesign(args.fname, args.dname, args.uniqID, args.group)

    # Get a list of samples to process, if processOnly is specified only analyze specified group.
    if args.processOnly:
        dat.design = dat.design[dat.design[args.group].isin(args.processOnly)]
        toProcess = dat.design.index
        dat.sampleIDs = toProcess.tolist()

    # Create dataframe with sampleIDs that are to be analyzed.
    #: :type wide: pandas.DataFrame
    dat.keep_sample(dat.sampleIDs)

    # Get list of pairwise combinations. If group is specified, only do within group combinations.
    combos = list()
    if args.group:
        # If group is given, only do within group pairwise combinations
        logger.info('Only doing within group, pairwise comparisons.')
        for groupName, dfGroup in dat.design.groupby(dat.group):
            combos.extend(list(combinations(dfGroup.index, 2)))
    else:
        logger.info('Doing all pairwise comparisons. This could take a while!')
        # Get all pairwise combinations for all samples
        combos.extend(list(combinations(dat.sampleIDs, 2)))

    # Open a multiple page PDF for plots
    ppBA = PdfPages(args.baName)

    # Loop over combinations and generate plots and return a list of flags.
    logger.info('Generating flags and plots.')
    flags = map(lambda combo: iterateCombo(dat, combo, ppBA), combos)

    # Close PDF with plots
    ppBA.close()

    # Merge flags
    logger.info('Merging outlier flags.')
    merged = Flags.merge(flags)

    # Summarize flags
    logger.info('Summarizing outlier flags.')
    propSample, propFeature = summarizeFlags(dat, merged, combos)
    plotFlagDist(propSample, propFeature, args.distName)

    # Create sample level flags
    flag_sample = Flags(index=dat.sampleIDs)
    flag_sample.addColumn(column='flag_sample_BA_outlier', mask=(propSample >= args.sampleCutoff))
    flag_sample.df_flags.to_csv(args.flagSample, sep='\t')

    # Create metabolite level flags
    flag_metabolite = Flags(dat.wide.index)
    flag_metabolite.addColumn(column='flag_feature_BA_outlier', mask=(propFeature >= args.featureCutoff))
    flag_metabolite.df_flags.to_csv(args.flagFeature, sep='\t')


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    # Expose cutoff as global
    global cutoff
    cutoff = args.residCutoff

    # Set up logging
    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
    else:
        sl.setLogger(logger)

    main(args)
