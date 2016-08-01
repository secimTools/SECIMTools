#!/usr/bin/env python

# Built-in packages
import logging
import argparse
from argparse import RawDescriptionHelpFormatter
from itertools import combinations
from collections import defaultdict

# Add-on packages
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
    parser.add_argument("--group", dest="group", action='store', required=True, help="Group/treatment identifier in design file.")
    parser.add_argument("--out", dest="oname", action='store', required=True, help="Output file name.")
    parser.add_argument("--fig", dest="ofig", action='store', required=True, help="Output figure name for q-q plots [pdf].")
    parser.add_argument("--fig2", dest="ofig2", action='store', required=True, help="Output figure name for volcano plots [pdf].")
    parser.add_argument("--debug", dest="debug", action='store_true', required=False, help="Add debugging log output.")
    args = parser.parse_args()

    return(args)


def createCbn(dat):
    """ Create all pairwise combinations of treatments

    :Arguments:
        :type dat: interface.wideToDesign
        :param dat: A wideToDesign object that will be used to get sample and
            feature information.

    :Returns:
        :rtype: dictionary
        :return: A dictionary of dictionaries with all possible pairwise
            combinations. Used this to create the various column headers in the
            results table.

    """

    combo = defaultdict(dict)
    cbn = list(combinations(range(0, len(dat.levels)), 2))
    for c1, c2 in cbn:
        n1 = dat.levels[c1]
        n2 = dat.levels[c2]
        combo[(c1, c2)]['Name'] = '{0} - {1}'.format(n1, n2)
        combo[(c1, c2)]['Levels'] = [n1, n2]
        combo[(c1, c2)]['Diff'] = 'Diff of treatment = {0}'.format(combo[(c1, c2)]['Name'])
        combo[(c1, c2)]['StdErr'] = 'StdError for Diff of treatment = {0}'.format(combo[(c1, c2)]['Name'])
        combo[(c1, c2)]['Tval'] = 't Value for Diff of treatment = {0}'.format(combo[(c1, c2)]['Name'])
        combo[(c1, c2)]['pTval'] = 'Prob>|t| for Diff of treatment = {0}'.format(combo[(c1, c2)]['Name'])
        combo[(c1, c2)]['lpTval'] = '-log10(p-value) for Diff of treatment = {0}'.format(combo[(c1, c2)]['Name'])
        combo[(c1, c2)]['sTval'] = 'Sig Index for Diff of treatment = {0}'.format(combo[(c1, c2)]['Name'])
    return combo


def initResults(dat):
    """ Initialize the results dataset

    :Arguments:
        :type dat: interface.wideToDesign
        :param dat: A wideToDesign object that will be used to get sample and
            feature information.

    :Returns:
        :rtype: pandas.DataFrame
        :returns: Initializes the results table, adding a row for each
            compound and all column headers.

    """
    base = ['GrandMean', 'ErrorSS']
    means = ['Mean treatment {0}'.format(x) for x in dat.levels]
    base2 = ['ErrorSS', 'ModelSS', 'TotalSS', 'MSE', 'NDF_treatment', 'DDF_treatment',
             'SampleVariance', 'RSquare', 'F_T3_treatment',
             'PrF_T3_treatment']

    combo = createCbn(dat)
    diff = [combo[x]['Diff'] for x in sorted(combo.keys())]
    SE = [combo[x]['StdErr'] for x in sorted(combo.keys())]
    T = [combo[x]['Tval'] for x in sorted(combo.keys())]
    pT = [combo[x]['pTval'] for x in sorted(combo.keys())]
    lpT = [combo[x]['lpTval'] for x in sorted(combo.keys())]
    spT = [combo[x]['sTval'] for x in sorted(combo.keys())]

    cols = base + means + base2 + diff + SE + T + pT + lpT + spT
    return pd.DataFrame(index=dat.wide.index, columns=cols)


def oneWay(dat, compound, results):
    """ One-way ANOVA

    Run a simple one-way anova, where compound is the dependent variable and
    group is the independent variable.

    :Arguments:
        :type dat: interface.wideToDesign
        :param dat: A wideToDesign object that will be used to get sample and
            feature information.

        :type compound: string
        :param compound: compound (str): The name of the current compound.

        :type results: pandas.DataFrame
        :param results: The results table, corresponding values will be
            replaced.

    :Returns:
        :rtype: pandas.DataFrame
        :returns: Updates in place the results table. Adds 'Anova
            model:...', 'F_T3_treatment', 'PrF_T3_treatment', 'ErrorSS', 'ModelSS',
            'TotalSS', 'MSE', 'NDF_treatment', 'DDF_treatment', 'SampleVariance',
            'RSquare' to results.

        :rtype: pandas.Series
        :returns: Pearson normalized residuals. (residuals / sqrt(MSE))

    """
    formula = '{0} ~ {1}'.format(str(compound), str(dat.group))
    model_lm = ols(formula, data=dat.trans).fit()
    results.ix[compound, 'F_T3_treatment'] = model_lm.fvalue
    results.ix[compound, 'PrF_T3_treatment'] = model_lm.f_pvalue
    results.ix[compound, 'ErrorSS'] = model_lm.ssr
    results.ix[compound, 'ModelSS'] = model_lm.ess
    results.ix[compound, 'TotalSS'] = model_lm.ess + model_lm.ssr
    results.ix[compound, 'MSE'] = model_lm.mse_resid
    results.ix[compound, 'NDF_treatment'] = int(model_lm.df_model)
    results.ix[compound, 'DDF_treatment'] = int(model_lm.df_resid)
    results.ix[compound, 'SampleVariance'] = dat.trans[compound].var()
    results.ix[compound, 'RSquare'] = model_lm.rsquared
    resid = model_lm.resid / np.sqrt(model_lm.mse_resid)
    fitted = model_lm.fittedvalues
    return pd.Series(resid, name=compound), pd.Series(fitted, name=compound)


def calcDiff(dat, compound, grpMeans, combo, results):
    """ Calculate group means and the differences between group means.

    :Arguments:
        :type dat: interface.wideToDesign
        :param dat: A wideToDesign object that will be used to get sample and
            feature information.

        :type compound: string
        :param compound: compound (str): The name of the current compound.

        :type grpMeans: pandas.DataFrame
        :param grpMeans: A table with the overall mean for each group.

        :type combo: dictionary
        :param combo: A dictionary of dictionaries with all possible pairwise
            combinations. Used this to create the various column headers in the
            results table.

        :type results: pandas.DataFrame
        :param results: The results table, corresponding values will be
        replaced.

    :Returns:
        :rtype: pandas.DataFrame
        :returns: Updates the results table. Adds 'Diff' to results.

    """
    for lvl in dat.levels:
        results.ix[compound, 'Mean treatment {0}'.format(lvl)] = grpMeans.ix[compound, lvl]

    for key in sorted(combo.keys()):
        results.ix[compound, combo[key]['Diff']] = grpMeans.ix[compound, combo[key]['Levels'][0]] - grpMeans.ix[compound, combo[key]['Levels'][1]]


def calcDiffSE(dat, compound, combo, results):
    """ Calculate the Standard Error between differences.

    :Arguments:
        :type dat: interface.wideToDesign
        :param dat: A wideToDesign object that will be used to get sample and
            feature information.

        :type compound: string
        :param compound: compound (str): The name of the current compound.

        :type combo: dictionary
        :param combo: A dictionary of dictionaries with all possible pairwise
            combinations. Used this to create the various column headers in the
            results table.

        :type results: pandas.DataFrame
        :param results: The results table, corresponding values will be
        replaced.

    :Returns:
        :rtype: pandas.DataFrame
        :returns: Updated results table. Adds 'StdErr' to results.

    """
    # MSE from ANOVA
    MSE = results.ix[compound, 'MSE']

    # Group by Treatment, need number of samples per treatment
    grp = dat.design.groupby(dat.group)

    for cbn in combo.values():
        n1 = cbn['Levels'][0]
        n2 = cbn['Levels'][1]
        df1 = len(grp.get_group(n1))
        df2 = len(grp.get_group(n2))
        SE = np.sqrt((MSE / df1) + (MSE / df2))
        results.ix[compound, cbn['StdErr']] = SE


def tTest(compound, combo, results, cutoff=4):
    """ Calculate T-value and T-critical.

    :Arguments:
        :type compound: string
        :param compound: compound (str): The name of the current compound.

        :type combo: dictionary
        :param combo: A dictionary of dictionaries with all possible pairwise

        :type results: pandas.DataFrame
        :param results: The results table, corresponding values will be

        :type cutoff: int
        :param cutoff: The cutoff value for significance [default: 4].

    :Returns:
        :rtype: pandas.DataFrame
        :returns: Updated results table. Adds 'Tval', 'pTval', 'lpTval',
                  'sTval' to results.

    """
    for key in sorted(combo.keys()):
        results.ix[compound, combo[key]['Tval']] = results.ix[compound, combo[key]['Diff']] / results.ix[compound, combo[key]['StdErr']]
        results.ix[compound, combo[key]['pTval']] = stats.t.sf(abs(results.ix[compound, combo[key]['Tval']]), results.ix[compound, 'DDF_treatment']) * 2
        results.ix[compound, combo[key]['lpTval']] = -np.log10(results.ix[compound, combo[key]['pTval']])
        if np.abs(results.ix[compound, combo[key]['lpTval']]) > cutoff:
            results.ix[compound, combo[key]['sTval']] = 1
        else:
            results.ix[compound, combo[key]['sTval']] = 0


def qqPlot(resids, fit, oname):
    """ Plot the residual diagnostic plots by sample.

    Output q-q plot, boxplots and distributions of the residuals. These plots
    will be used diagnose if residuals are approximately normal.

    :Arguments:
        :type resids: pandas.Series
        :param resids: Pearson normalized residuals. (residuals / sqrt(MSE))

        :type oname: string
        :param oname: Name of the output file in pdf format.

    :Returns:
        :rtype: PDF
        :returns: Outputs a pdf file containing all plots.

    """
    with PdfPages(oname) as pdf:
        tresid = resids.T
        tfit = fit.T
        for col in tresid.columns:
            fig = plt.figure(figsize=(8, 8))
            fig.suptitle(col)

            # Set up layout using gridspec
            gs = gridspec.GridSpec(2, 3, wspace=0)
            ax1 = plt.subplot(gs[0, 0])
            ax2 = plt.subplot(gs[0, 1])
            ax3 = plt.subplot(gs[0, 2], sharey=ax1)
            ax4 = plt.subplot(gs[1, :])

            # Generate Plots
            sm.graphics.qqplot(tresid[col], fit=True, line='r', ax=ax1)
            ax2.boxplot(tresid[col].values.ravel(), vert=True)
            ax3.hist(tresid[col].values.ravel(), orientation='horizontal')
            ax4.scatter(tfit[col], tresid[col])
            ax4.axhline(0, c='r', lw=1)

            # Clean up plots
            ## Set titles
            ax2.set_title('Distribution of P-values')
            ax4.set_title('Fitted Values vs Residuals')

            ## Add axis labels
            ax4.set_xlabel('Fitted Values')
            ax4.set_ylabel('Standardized Residuals')

            ## remove unnecessary axis
            ax1.get_xaxis().set_visible(False)
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
            ax3.get_xaxis().set_visible(False)
            ax3.get_yaxis().set_visible(False)

            pdf.savefig(fig)
            plt.close(fig)


def volcano(combo, results, oname, cutoff=4):
    """ Plot volcano plots.

    Creates volcano plots to compare means, for all pairwise differences.

    :Arguments:

        :type combo: dictionary
        :param combo: A dictionary of dictionaries with all possible pairwise
            combinations. Used this to create the various column headers in the
            results table.

        :type oname: string
        :param oname: Name of the output file in pdf format.

        :type cutoff: int
        :param cutoff: The cutoff value for significance [default: 4].

    :Returns:
        :rtype: PD
        :returns: Outputs a pdf file containing all plots.

    """

    with PdfPages(oname) as pdf:
        for key in sorted(combo.keys()):
            diff = combo[key]['Diff']
            pval = combo[key]['lpTval']

            # Set up figure
            fig = plt.figure(figsize=(8, 8))
            fig.suptitle(combo[key]['Name'])
            ax = fig.add_subplot(111)

            # Plot all results
            results.plot(x=diff, y=pval, kind='scatter', ax=ax)

            # Color results beyond threshold red
            subset = results[results[pval] > cutoff]
            if not subset.empty:
                subset.plot(x=diff, y=pval, kind='scatter', color='r', ax=ax)

            plt.axhline(cutoff, color='r', ls='--', lw=2)
            pdf.savefig(fig)
            plt.close(fig)


def main(args):
    # Import data
    logger.info(u'Importing data with following parameters: \n\tWide: {0}\n\tDesign: {1}\n\tUnique ID: {2}\n\tGroup Column: {3}'.format(args.fname, args.dname, args.uniqID, args.group))
    dat = wideToDesign(args.fname, args.dname, args.uniqID, args.group, clean_string=True)

    # Drop missing values
    dat.wide = dat.wide.dropna()

    results = initResults(dat)


    # Transpose the data
    dat.trans = dat.transpose()

    # Group by Treatment
    grp = dat.trans.groupby(dat.group)
    grpMeans = grp.mean().T
    combo = createCbn(dat)

    resids = list()
    fitted = list()
    # Iterate over compound
    logger.info('Running row-by-row analysis.')
    for compound in dat.wide.index.tolist():
        # Get Overall Mean
        results.ix[compound, 'GrandMean'] = dat.trans[compound].mean()

        # run one-way ANOVA
        resid, fit = oneWay(dat, compound, results)
        resids.append(resid)
        fitted.append(fit)

        # Calculate mean differences
        calcDiff(dat, compound, grpMeans, combo, results)

        # Calculate SE of difference between means
        calcDiffSE(dat, compound, combo, results)

        # Calculate T-test
        tTest(compound, combo, results)

    residDat = pd.concat(resids, axis=1)
    fitDat = pd.concat(fitted, axis=1)

    # Generate qqplots
    logger.info('Generating q-q plots.')
    qqPlot(residDat, fitDat, args.ofig)

    # Generate Volcano plots
    logger.info('Generating volcano plots.')
    volcano(combo, results, args.ofig2)

    # write results table
    results = results.convert_objects(convert_numeric=True)
    results.index = pd.Series([dat.revertStr(x) for x in results.index])
    results = results.apply(lambda x: x.round(4))
    results.to_csv(args.oname, sep="\t")


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
    else:
        sl.setLogger(logger)

    main(args)
