#!/usr/bin/env python
################################################################################
# DATE: 2017/03/09
#
# SCRIPT: bland_altmant_plot.py
#
# VERSION: 1.3
#
# AUTHORS: Miguel A Ibarra <miguelib@ufl.edu>
#          Matt Thoburn <mthoburn@ufl.edu>
#          Oleksandr Moskalenko <om@rc.ufl.edu>
#
# DESCRIPTION: This script takes a a wide format file and makes a Bland-Altman plot
#
# The output is a set of graphs and spreadsheets of flags
#
################################################################################
# Import future libraries


# Import built-in libraries
import os
import logging
import argparse
from itertools import combinations
from argparse import RawDescriptionHelpFormatter

# Import add-on libraries
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# Import local data libraries
from secimtools.dataManager import logger as sl
from secimtools.dataManager.flags import Flags
from secimtools.dataManager.interface import wideToDesign

# Import local plotting libraries
from secimtools.visualManager import module_bar as bar
from secimtools.visualManager import module_lines as lines
from secimtools.visualManager import module_scatter as scatter
from secimtools.visualManager.manager_color import colorHandler
from secimtools.visualManager.manager_figure import figureHandler

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
    parser = argparse.ArgumentParser(description=description,
            formatter_class=RawDescriptionHelpFormatter)
    # Standard Input
    Standard = parser.add_argument_group(title='Standard input',
            description='Standard input for SECIM tools.')
    Standard.add_argument('-i',"--input", dest="input", action='store',
            required=True, help="Input dataset in wide format.")
    Standard.add_argument('-d',"--design", dest="design", action='store',
            required=True, help="Design file.")
    Standard.add_argument('-id',"--ID", dest="uniqID", action='store',
            required=True, help="Name of the column with unique identifiers.")
    Standard.add_argument('-g',"--group", dest="group", action='store',
            required=False, help="Group/treatment identifier in design file"\
            " [Optional].")
    # Tool output
    output = parser.add_argument_group(title='Required input',
            description='Additional required input for this tool.')
    output.add_argument('-f',"--figure", dest="baName", action='store',
            required=True, help="Name of the output PDF for Bland-Altman plots.")
    output.add_argument('-fd',"--flag_dist", dest="distName", action='store',
            required=True, help="Name of the output TSV for distribution flags.")
    output.add_argument('-fs',"--flag_sample", dest="flagSample", action='store',
            required=True, help="Name of the output TSV for sample flags.")
    output.add_argument('-ff',"--flag_feature", dest="flagFeature", action='store',
            required=True, help="Name of the output TSV for feature flags.")
    ## AMM added following 2 arguments
    output.add_argument('-pf',"--prop_feature", dest="propFeature", action='store',
            required=True, help="Name of the output TSV for proportion of features.")
    output.add_argument('-ps',"--prop_sample", dest="propSample", action='store',
            required=True, help="Name of the output TSV for proportion of samples.")


    # Tool Input
    tool = parser.add_argument_group(title='Optional Settings')
    tool.add_argument('-po',"--process_only", dest="processOnly",
            action='store', nargs='+', default=False, required=False,
            help="Only process the given groups (list groups separated by"\
            " spaces) [Optional].")
    tool.add_argument('-rc',"--resid_cutoff", dest="residCutoff",
            action='store', default=3, type=int, required=False,
            help="Cutoff value for flagging outliers [default=3].")

    tool.add_argument('-sfc',"--sample_flag_cutoff", dest="sampleCutoff",
            action='store', default=.20, type=float, required=False,
            help="Proportion cutoff value when flagging samples [default=0.20].")
    tool.add_argument('-ffc',"--feature_flag_cutoff", dest="featureCutoff",
            action='store', default=.05, type=float, required=False,
            help="Proportion cutoff value when flagging features [default=0.05].")

    group4 = parser.add_argument_group(title='Development Settings')
    group4.add_argument("--debug", dest="debug", action='store_true',
            required=False, help="Add debugging log output.")
    args = parser.parse_args()
    # Check if sample cutoff is within 0-1
    if (args.sampleCutoff > 1) | (args.sampleCutoff < 0):
        parser.error('sample_flag_cutoff must be a number between 0 and 1')
    if (args.featureCutoff > 1) | (args.featureCutoff < 0):
        parser.error('feature_flag_cutoff must be a number between 0 and 1')

    # Standardize paths
    args.input       = os.path.abspath(args.input)
    args.design      = os.path.abspath(args.design)
    args.baName      = os.path.abspath(args.baName)
    args.distName    = os.path.abspath(args.distName)
    args.flagSample  = os.path.abspath(args.flagSample)
    args.flagFeature = os.path.abspath(args.flagFeature)
    # AMM added following 2
    args.propFeature = os.path.abspath(args.propFeature)
    args.propSamplee = os.path.abspath(args.propSample)

    return args

def summarizeFlags(dat, flags, combos):
    """ Given a set of flags calculate the proportion of times a feature is
        flagged.

    :Arguments:
        :type dat: interface.wideToDesign
        :param dat: A wideToDesign object that will be used to get sample and
            feature information.

        :type flags: pandas.DataFrame
        :param flags: A Dataframe of flags for all pairwise comparisons.

        :type combos: list
        :param combos: combinations of flags

    :Returns:
        :rtype: tuple of pandas.DataFrame
        :return: Two DataFrames, the first has the proportion of samples that
            were flagged. The second has the proportion of features flagged.

    """
    # Create a data frame that is the same dimensions as wide. Where each cell
    # will be the sum of flags.
    flagSum = pd.DataFrame(index=flags.index, columns=dat.wide.columns)
    flagSum.fillna(0, inplace=True)

    # Create a data frame that is the same dimensions as wide. Where each cell
    # is the total number of comparisons.
    flagTotal = pd.DataFrame(index=flags.index, columns=dat.wide.columns)
    flagTotal.fillna(0, inplace=True)

    # Create a data frame that is the same dimensions as wide. Where each cell
    # will be the sum of flags.
    flagSum_p = pd.DataFrame(index=flags.index, columns=dat.wide.columns)
    flagSum_p.fillna(0, inplace=True)

    # Create a data frame that is the same dimensions as wide. Where each cell
    # is the total number of comparisons.
    flagTotal_p = pd.DataFrame(index=flags.index, columns=dat.wide.columns)
    flagTotal_p.fillna(0, inplace=True)

    # Create a data frame that is the same dimensions as wide. Where each cell
    # will be the sum of flags.
    flagSum_c = pd.DataFrame(index=flags.index, columns=dat.wide.columns)
    flagSum_c.fillna(0, inplace=True)

    # Create a data frame that is the same dimensions as wide. Where each cell
    # is the total number of comparisons.
    flagTotal_c = pd.DataFrame(index=flags.index, columns=dat.wide.columns)
    flagTotal_c.fillna(0, inplace=True)

    # Create a data frame that is the same dimensions as wide. Where each cell
    # will be the sum of flags.
    flagSum_d = pd.DataFrame(index=flags.index, columns=dat.wide.columns)
    flagSum_d.fillna(0, inplace=True)

    # Create a data frame that is the same dimensions as wide. Where each cell
    # is the total number of comparisons.
    flagTotal_d = pd.DataFrame(index=flags.index, columns=dat.wide.columns)
    flagTotal_d.fillna(0, inplace=True)

    for sampleID in dat.sampleIDs:
        # Get list of flags that contain the current sampleID
        flagList = ["flag_{0}_{1}".format(c[0], c[1]) for c in combos if sampleID in c]
        flagList_p = ["flag_pearson_{0}_{1}".format(c[0], c[1]) for c in combos if sampleID in c]
        flagList_c = ["flag_cooks_{0}_{1}".format(c[0], c[1]) for c in combos if sampleID in c]
        flagList_d = ["flag_dffits_{0}_{1}".format(c[0], c[1]) for c in combos if sampleID in c]

        # Sum the flags in flags for the current sampleID
        flagSum.loc[:, sampleID] = flags[flagList].sum(axis=1).values
        flagSum_p.loc[:, sampleID] = flags[flagList_p].sum(axis=1).values
        flagSum_c.loc[:, sampleID] = flags[flagList_c].sum(axis=1).values
        flagSum_d.loc[:, sampleID] = flags[flagList_d].sum(axis=1).values

        # Get the totals of possible flags in flags for the current sampleID
        flagTotal.loc[:, sampleID] = flags[flagList].count(axis=1).values
        flagTotal_p.loc[:, sampleID] = flags[flagList_p].count(axis=1).values
        flagTotal_c.loc[:, sampleID] = flags[flagList_c].count(axis=1).values
        flagTotal_d.loc[:, sampleID] = flags[flagList_d].count(axis=1).values

    # Calculate the proportion of samples and features using the marginal sums.
    propSample = flagSum.sum(axis=0) / flagTotal.sum(axis=0)
    propFeature = flagSum.sum(axis=1) / flagTotal.sum(axis=1)

    # Calculate the proportion of samples and features using the marginal sums.
    propSample_p = flagSum_p.sum(axis=0) / flagTotal_p.sum(axis=0)
    propFeature_p = flagSum_p.sum(axis=1) / flagTotal_p.sum(axis=1)

    # Calculate the proportion of samples and features using the marginal sums.
    propSample_c = flagSum_c.sum(axis=0) / flagTotal_c.sum(axis=0)
    propFeature_c = flagSum_c.sum(axis=1) / flagTotal_c.sum(axis=1)

    # Calculate the proportion of samples and features using the marginal sums.
    propSample_d = flagSum_d.sum(axis=0) / flagTotal_d.sum(axis=0)
    propFeature_d = flagSum_d.sum(axis=1) / flagTotal_d.sum(axis=1)

    return propSample, propFeature, propSample_p, propFeature_p, propSample_c, propFeature_c, propSample_d, propFeature_d

def plotFlagDist(propSample, propFeature, pdf):
    """
    Plot the distribution of proportion of samples and features that
    were outliers.

    :Arguments:
        :type propSample: pandas.DataFrame
        :param propSample: Data frame of the proportion of samples flagged as
            an outlier.

        :type propFeature: pandas.DataFrame
        :param propFeature: Data frame of the proportion of features flagged as
            an outlier.

        :type pdf: string
        :param pdf: Filename of pdf to save plots.

    :Returns:
        :rtype: matplotlib.backends.backend_pdf.PdfPages
        :returns: Saves two bar plots to pdf.

    """
    # sort samples
    propSample.sort_values(inplace=True,ascending=False)

    # sort compounds
    propFeature.sort_values(inplace=True,ascending=False)

    # Make Plots
    ## Open pdf for plotting
    ppFlag = PdfPages(pdf)

    # Open figure handler instance
    fh = figureHandler(proj='2d')
    keys = list(propSample.head(30).keys())

    # Plotting quickBar
    #breakpoint()
    bar.quickBar(ax=fh.ax[0],y=list(propSample.head(30)),x=keys)

    # Formating axis
    fh.formatAxis(xlim=(0,len(keys) + 1), ylim="ignore", xTitle="Sample ID",
                yTitle="Proportion of features that were outliers.")

    # Save Figure in PDF
    ppFlag.savefig(fh.fig, bbox_inches='tight')

    ## Plot samples
    # Open figure handler instance
    fh = figureHandler(proj='2d')
    keys = list(propFeature.head(30).keys())

    # Plot bar plot
    bar.quickBar(ax=fh.ax[0],y=list(propFeature.head(30)),x=keys)

    # Format Axis
    fh.formatAxis(xlim=(0,len(keys) + 1), ylim="ignore", xTitle="Feature ID",
        yTitle="Proportion of samples that a feature was an outlier.")

    # Plot samples
    ppFlag.savefig(fh.fig, bbox_inches="tight")

    ## Close pdf
    ppFlag.close()

def buildTitle(dat, xName, yName):
    """ Build plot title.

    :Arguments:
        :type dat: interface.wideToDesign
        :param dat: A wide to design object

        :type xName: string
        :param xName: String containing the sampleID for x

        :type yName: string
        :param yName: String containing the sampleID for y

    :Returns:
        :rtype: string
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
    y=y.apply(float)
    x=x.apply(float)
    model = sm.OLS(y, x, missing='drop')
    results = model.fit()

    # Get fit and influence stats
    fitted = results.fittedvalues
    infl = results.get_influence()
    CD = infl.cooks_distance
    (dffits, thresh) = infl.dffits
    DF = abs(dffits) > thresh

    influence = pd.DataFrame({'cooksD': CD[0], 'cooks_pval': CD[1], 'dffits': DF},
                 index=fitted.index)

    # Get Residuals
    presid = pd.Series(results.resid_pearson, index=results.resid.index)
    resid = pd.concat([results.resid, presid], axis=1)
    resid.columns = pd.Index(['resid', 'resid_pearson'])

    # Get 95% CI
    prstd, lower, upper = wls_prediction_std(results)

    return lower, upper, fitted, resid, influence

def makeBA(x, y, ax, fh):
    """ Function to make BA Plot comparing x vs y.

    :Arguments:
        :type x: pandas.Series
        :param x: Series of first sample, treated as independent variable.

        :type y: pandas.Series
        :param y: Series of second sample, treated as dependent variables.

        :type ax: matplotlib.axis
        :param ax: Axis which to plot.

        :type fh: figureHandler
        :param fh: figure to draw BA plots onto.

    :Returns:
        :rtype: pandas.Series
        :returns: A Series containing Boolean values with True
              indicating a value is more extreme than CI and False indicating a
              value falls inside CI.

    """
    # Make BA plot
    x = x.apply(float)
    y = y.apply(float)

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
    mask  = mask1 | mask2 | mask3

    # Create BA plot
    scatter.scatter2D(ax=ax, x=mean[~mask], y=diff[~mask],colorList='b')
    scatter.scatter2D(ax=ax, x=mean[mask],  y=diff[mask], colorList='r')

    # Plot regression lines
    ax.plot(mean, lower, 'r:')
    ax.plot(mean, fitted, 'r')
    ax.axhline(0, color='k')
    ax.plot(mean, upper, 'r:')

    #Adjust axes
    fh.formatAxis(axnum=1,xlim='ignore',ylim='ignore',axTitle='Bland-Altman Plot',
        xTitle='Mean\n{0} & {1}'.format(x.name, y.name),
        yTitle='Difference\n{0} - {1}'.format(x.name, y.name),grid=False)

    return mask, mask1, mask2, mask3

def makeScatter(x, y, ax, fh):
    """ Plot a scatter plot of x vs y.

    :Arguments:
        :type x: pandas.Series
        :param x: Series of first sample, treated as independent variable.

        :type y: pandas.Series
        :param y: Series of second sample, treated as dependent variables.

        :type ax: matplotlib.axis
        :param ax: Axis which to plot.

        :type fh: figureHandler
        :param fh: figure to draw BA plots onto.

    :Returns:
        :rtype: matplotlib.axis
        :returns: A matplotlib axis with a scatter plot.

    """
    #logger.info('{0}, {1}'.format(x.name, y.name))
    # Get Upper and Lower CI from regression
    lower, upper, fitted, resid, infl = runRegression(x, y)

    # Plot scatter
    scatter.scatter2D(x=x,y=y,ax=ax,colorList = list("b"))
    # Plot regression lines
    # If there are missing data, x and the result vectors won't have the same
    # dimensions. First filter x by the index of the fitted values then plot.
    x2 = x.loc[fitted.index]
    lines.drawCutoff(x=x2,y=lower,ax=ax)
    lines.drawCutoff(x=x2,y=fitted,ax=ax)
    lines.drawCutoff(x=x2,y=upper,ax=ax)
    # Adjust plot
    fh.formatAxis(axnum=0,xTitle=x.name,yTitle=y.name,axTitle='Scatter plot',grid=False)

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
    fh = figureHandler(proj='2d',numAx=2,numRow=2,numCol=2,arrangement=[(0,0,1,2),(0,1,1,2)])

    # Scatter Plot of c1 vs c2
    makeScatter(dat.wide.loc[:, c1], dat.wide.loc[:, c2], fh.ax[0],fh)

    # BA plot of c1 vs c2
    outlier, pearson, cooks, dffits = makeBA(dat.wide.loc[:, c1], dat.wide.loc[:, c2], fh.ax[1],fh)

    # Build plot title
    title = buildTitle(dat, c1, c2)

    # Add plot title to the figure
    fh.formatAxis(figTitle=title)

    # Stablishing a tight layout for the figure
    plt.tight_layout(pad=2,w_pad=.05)

    # Shinking figure
    fh.shrink(top=.85,bottom=.25,left=.15,right=.9)

    # Output figure to pdf
    fh.addToPdf(dpi=90,pdfPages=pdf)

    # Create flags
    flag = Flags(index=dat.wide.index)
    flag.addColumn(column='flag_{0}_{1}'.format(c1, c2), mask=outlier)
    flag.addColumn(column='flag_pearson_{0}_{1}'.format(c1, c2), mask=pearson)
    flag.addColumn(column='flag_cooks_{0}_{1}'.format(c1, c2), mask=cooks)
    flag.addColumn(column='flag_dffits_{0}_{1}'.format(c1, c2), mask=dffits)

    return flag.df_flags

def main(args):
    # Import data
    dat = wideToDesign(args.input, args.design, args.uniqID, args.group,
                        logger=logger)

    # Get a list of samples to process, if processOnly is specified only
    # analyze specified group.
    if args.processOnly:
        dat.design = dat.design[dat.design[args.group].isin(args.processOnly)]
        toProcess = dat.design.index
        dat.sampleIDs = toProcess.tolist()

    # Create dataframe with sampleIDs that are to be analyzed.
    dat.keep_sample(dat.sampleIDs)

    # Get list of pairwise combinations. If group is specified, only do
    # within group combinations.
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
    flags = [iterateCombo(dat, combo, ppBA) for combo in combos]

    # Close PDF with plots
    ppBA.close()

    # Merge flags
    logger.info('Merging outlier flags.')
    merged = Flags.merge(flags)

    # Summarize flags
    logger.info('Summarizing outlier flags.')
    propSample, propFeature, propSample_p, propFeature_p, propSample_c, propFeature_c, propSample_d, propFeature_d = summarizeFlags(dat, merged, combos)
    plotFlagDist(propSample, propFeature, args.distName)

    ## AMM - output sample and feature proportions
    propSample.to_csv(args.propSample, sep='\t')
    propFeature.to_csv(args.propFeature, sep='\t')

    # Create sample level flags
    flag_sample = Flags(index=dat.sampleIDs)
    flag_sample.addColumn(column='flag_sample_BA_outlier',
                        mask=(propSample >= args.sampleCutoff))
    flag_sample.addColumn(column='flag_sample_BA_pearson',
                        mask=(propSample_p >= args.sampleCutoff))
    flag_sample.addColumn(column='flag_sample_BA_cooks',
                        mask=(propSample_c >= args.sampleCutoff))
    flag_sample.addColumn(column='flag_sample_BA_dffits',
                        mask=(propSample_d >= args.sampleCutoff))
    flag_sample.df_flags.index.name = "sampleID"
    flag_sample.df_flags.to_csv(args.flagSample, sep='\t')

    # Create metabolite level flags
    flag_metabolite = Flags(dat.wide.index)
    flag_metabolite.addColumn(column='flag_feature_BA_outlier',
                        mask=(propFeature >= args.featureCutoff))
    flag_metabolite.addColumn(column='flag_feature_BA_pearson',
                        mask=(propFeature_p >= args.featureCutoff))
    flag_metabolite.addColumn(column='flag_feature_BA_cooks',
                        mask=(propFeature_c >= args.featureCutoff))
    flag_metabolite.addColumn(column='flag_feature_BA_dffits',
                        mask=(propFeature_d >= args.featureCutoff))
    flag_metabolite.df_flags.to_csv(args.flagFeature, sep='\t')

    # Finish Script
    logger.info("Script Complete!")


if __name__ == '__main__':
    args = getOptions()
    global cutoff
    cutoff = args.residCutoff
    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
    else:
        sl.setLogger(logger)
    logger.info('Importing Data')
    main(args)
