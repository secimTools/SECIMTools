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
    parser.add_argument("--std", dest="std", action='store', choices=['STD', 'MEAN'], required=False, default=None, help="Group/treatment identifiers. Can be multiple names separated by a space.")
    parser.add_argument("--out", dest="oname", action='store', required=True, help="Output file name.")
    parser.add_argument("--fig", dest="ofig", action='store', required=True, help="Output figure name for q-q plots [pdf].")
    parser.add_argument("--fig2", dest="ofig2", action='store', required=True, help="Output figure name for volcano plots [pdf].")
    parser.add_argument("--debug", dest="debug", action='store_true', required=False, help="Add debugging log output.")
#     args = parser.parse_args()
    args = parser.parse_args(['--input', '/home/jfear/sandbox/secim/data/small.tsv',
                              '--design', '/home/jfear/sandbox/secim/data/ST000015_design_v2.tsv',
                              '--ID', 'Name',
                              '--group', 'treatment',
                              '--out', '/home/jfear/sandbox/secim/data/test.csv',
                              '--fig', '/home/jfear/sandbox/secim/data/test.pdf',
                              '--fig2', '/home/jfear/sandbox/secim/data/test2.pdf',
                              '--debug'])
    return(args)


def createCbn(dat):
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
    """ Initialize the results dataset """
    base = ['GrandMean', 'ErrorSS']
    means = ['Mean treatment {0}'.format(x) for x in dat.levels]
    base2 = ['ErrorSS', 'ModelSS', 'TotalSS', 'MSE', 'NDF_treatment', 'DDF_treatment',
             'Variance_Residual', 'SampleVariance', 'RSquare', 'F_T3_treatment',
             'PrF_T3_treatment', 'MeanMean', 'StdDevMean']
    means2 = ['Std Mean treatment {0}'.format(x) for x in dat.levels]

    combo = createCbn(dat)
    diff = [combo[x]['Diff'] for x in sorted(combo.keys())]
    SE = [combo[x]['StdErr'] for x in sorted(combo.keys())]
    T = [combo[x]['Tval'] for x in sorted(combo.keys())]
    pT = [combo[x]['pTval'] for x in sorted(combo.keys())]
    lpT = [combo[x]['lpTval'] for x in sorted(combo.keys())]
    spT = [combo[x]['sTval'] for x in sorted(combo.keys())]

    cols = base + means + base2 + means2 + diff + SE + T + pT + lpT + spT
    return pd.DataFrame(index=dat.wide.index, columns=cols)


def sasSTD(dat, results):
    """ Function standardize using SAS STD procedure """
    # Get transposed table
    trans = dat.transpose()

    # group by treatment
    grp = trans.groupby(dat.group)

    # Calculate group means and std
    meanMean = grp.mean().mean()
    stdMean = grp.mean().std()

    # Standardize using sas STD method
    standard = grp.transform(lambda x: (x - meanMean[x.name]) / stdMean[x.name])
    merged = standard.join(dat.design)
    merged.index.name = 'sampleID'

    # Add results to output table
    sMeans = merged.groupby(dat.group).mean().T
    for compound in dat.wide.index.tolist():
        results.ix[compound, 'MeanMean'] = meanMean.ix[compound]
        results.ix[compound, 'StdDevMean'] = stdMean.ix[compound]
        for lvl in dat.levels:
            results.ix[compound, 'Std Mean treatment {0}'.format(lvl)] = sMeans.ix[compound, lvl]

    return merged


def sasMEAN(dat, results):
    """ Function standardize using SAS STD procedure """
    # Get transposed table
    trans = dat.transpose()

    # group by treatment
    grp = trans.groupby(dat.group)

    # Calculate group means and std
    meanMean = grp.mean().mean()

    # Standardize using sas MEAN method
    standard = grp.transform(lambda x: x - meanMean[x.name])
    merged = standard.join(dat.design)
    merged.index.name = 'sampleID'

    # Add results to output table
    sMeans = merged.groupby(dat.group).mean().T
    for compound in dat.wide.index.tolist():
        results.ix[compound, 'MeanMean'] = meanMean.ix[compound]
        results.ix[compound, 'StdDevMean'] = 1
        for lvl in dat.levels:
            results.ix[compound, 'Std Mean treatment {0}'.format(lvl)] = sMeans.ix[compound, lvl]

    return merged


def oneWay(dat, compound, results):
    """ One-way ANOVA """
    formula = '{0} ~ {1}'.format(str(compound), str(dat.group))
    logger.info('ANOVA model: {0}'.format(formula))
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
    return pd.Series(resid, name=compound)


def calcDiff(dat, compound, grpMeans, combo, results):
    """ Calculate group means and the differences between group means """
    for lvl in dat.levels:
        results.ix[compound, 'Mean treatment {0}'.format(lvl)] = grpMeans.ix[compound, lvl]

    for key in sorted(combo.keys()):
        results.ix[compound, combo[key]['Diff']] = grpMeans.ix[compound, combo[key]['Levels'][0]] - grpMeans.ix[compound, combo[key]['Levels'][1]]


def calcDiffSE(dat, compound, combo, results):
    """ One-way ANOVA """

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
    """ Calculate T-value and T-critical """
    for key in sorted(combo.keys()):
        results.ix[compound, combo[key]['Tval']] = results.ix[compound, combo[key]['Diff']] / results.ix[compound, combo[key]['StdErr']]
        results.ix[compound, combo[key]['pTval']] = stats.t.sf(abs(results.ix[compound, combo[key]['Tval']]), results.ix[compound, 'DDF_treatment']) * 2
        results.ix[compound, combo[key]['lpTval']] = -np.log10(results.ix[compound, combo[key]['pTval']])
        if np.abs(results.ix[compound, combo[key]['lpTval']]) > cutoff:
            results.ix[compound, combo[key]['sTval']] = 1
        else:
            results.ix[compound, combo[key]['sTval']] = 0


def qqPlot(resids, oname):
    """ Plot the residuals by sample """
    with PdfPages(oname) as pdf:
        trans = resids.T
        for col in trans.columns:
            fig = plt.figure(figsize=(8, 8))
            fig.suptitle(col)
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)

            ax2.axes.get_xaxis().set_visible(False)
            ax2.axes.get_yaxis().set_visible(False)

            ax3 = fig.add_subplot(313)

            sm.graphics.qqplot(trans[col], fit=True, line='r', ax=ax1)
            ax2.boxplot(trans[col].values.ravel(), vert=False)
            ax3.hist(trans[col].values.ravel())

            pdf.savefig(fig)
            plt.close(fig)


def volcano(combo, results, oname, cutoff=4):
    """ Plot volcano plots """
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


def cleanCol(x, decimals=4):
    """ Round floats to `decimals` places """
    if isinstance(x, float):
        formatString = '%.{0}f'.format(str(decimals))
        x2 = float(formatString % x)
    else:
        x2 = x
    return x2


def main(args):
    # Import data
    logger.info('Importing data')
    dat = wideToDesign(args.fname, args.dname, args.uniqID, args.group, clean_string=True)
    results = initResults(dat)

    # Standardize the data
    if args.std == 'STD':
        logger.info('Calculating STD standardization')
        dat.sdat = sasSTD(dat, results)
    elif args.std == 'MEAN':
        logger.info('Calculating MEAN standardization')
        dat.sdat = sasMEAN(dat, results)

    # NOTE: JMP does not standardized values for the ANOVA. The standardized
    # values are only used for the creation of Hierarchical clusters, parallel
    # plots, and PCA.
    dat.trans = dat.transpose()

    # Group by Treatment
    grp = dat.trans.groupby(dat.group)
    grpMeans = grp.mean().T
    combo = createCbn(dat)

    resids = list()
    # Iterate over compound
    for compound in dat.wide.index.tolist():
        # Get Overall Mean
        results.ix[compound, 'GrandMean'] = dat.trans[compound].mean()
        # run one-way anova
        resid = oneWay(dat, compound, results)
        resids.append(resid)

        # Calculate mean differences
        calcDiff(dat, compound, grpMeans, combo, results)

        # Calculate SE of difference between means
        calcDiffSE(dat, compound, combo, results)

        # Calculate T-test
        tTest(compound, combo, results)

    residDat = pd.concat(resids, axis=1)

    # Generate qqplots
    qqPlot(residDat, args.ofig)

    # Generate Volcano plots
    volcano(combo, results, args.ofig2)

    # write results table
    #TODO: Test the reversion part of the class
    dat.revertSring(results)
    clean = results.applymap(lambda x: cleanCol(x))
    clean.to_csv(args.oname, sep="\t")


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
    else:
        sl.setLogger(logger)

    main(args)
