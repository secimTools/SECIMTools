#!/usr/bin/env python

# Built-in packages
from __future__ import division
import logging
import argparse
from argparse import RawDescriptionHelpFormatter

# Add-on packages
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# Local Packages
from interface import wideToDesign
import logger as sl


def getOptions():
    """ Function to pull in arguments """
    description = """ """
    parser = argparse.ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)
    group1 = parser.add_argument_group(title='Standard input', description='Standard input for SECIM tools.')
    group1.add_argument('--input', dest='fname', action='store', required=True, help="Input dataset in wide format.")
    group1.add_argument('--design', dest='dname', action='store', required=True, help="Design file.")
    group1.add_argument('--ID', dest='uniqID', action='store', required=True,
                        help="Name of the column with unique identifiers.")
    group1.add_argument('--group', dest='group', action='store', required=True,
                        help="Group/treatment identifier in design file [Optional].")
    group1.add_argument('--order', dest='order', action='store', required=True, help='')

    group2 = parser.add_argument_group(title='Required Output')
    group2.add_argument('--fig', dest='oname', action='store', required=True, help="Name of PDF to save scatter plots.")
    group2.add_argument('--table', dest='tname', action='store', required=True,
                        help="Name of table for scatter plots")

    group4 = parser.add_argument_group(title='Development Settings')
    group4.add_argument("--debug", dest="debug", action='store_true', required=False, help="Add debugging log output.")

    args = parser.parse_args()

    return (args)


def runOrder(col):
    """ Run order regression
    :Arguments:
        col (pd.Series): A Pandas series with run order ['run'] as the index
            and ['val'] as the value to regressed on.

    :Returns:
        name (str): Name of the column.

        X's (int): Values of X ['run']

        Y's (int): Values of Y ['val']

        results (sm.RegressionResults): Regression results.

    """
    # Drop missing values and make sure everything is numeric
    clean = col.dropna().reset_index().convert_objects(convert_numeric=True)
    clean.columns = ['run', 'val']

    if clean.shape[0] >= 5:
        # Fit model
        model = smf.ols(formula='val ~ run', data=clean, missing='drop')
        results = model.fit()
        return col.name, clean['run'], clean['val'], results
    else:
        logger.warn('{} had fewer than 5 samples without NaNs, it will be ignored.'.format(col.name))


def flagPval(val, alpha):
    if val <= alpha:
        flag = True
    else:
        flag = False

    return flag


def makeScatter(row, pdf):
    """ Plot a scatter plot of x vs y. """
    # Split out row results
    name, x, y, res = row

    # Get fitted values
    fitted = res.fittedvalues

    # Get slope and p-value
    slope = res.params['run']
    pval = res.pvalues['run']
    rsq = res.rsquared

    # Make Flag based on p-value
    flag_runOrder_pval_05 = flagPval(pval, 0.05)
    flag_runOrder_pval_01 = flagPval(pval, 0.01)

    # Make scatter plot if p-pvalue is less than 0.05
    if flag_runOrder_pval_05 == 1:
        # Get 95% CI
        prstd, lower, upper = wls_prediction_std(res)

        # Sort CIs for Plotting
        toPlot = pd.DataFrame({'x': x, 'lower': lower, 'upper': upper})
        toPlot.sort(columns='x', inplace=True)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.scatter(x, y)
        ax.plot(toPlot['x'], toPlot['lower'], 'r-')
        ax.plot(x, fitted, 'c-')
        ax.plot(toPlot['x'], toPlot['upper'], 'r-')
        ax.set_xlabel('Run Order')
        ax.set_ylabel('Value')
        ax.set_title(u'{}\nScatter plot'.format(name))
        ax.text(.7, .85, u'Slope: {0:.4f}\n(p-value = {1:.4f})\nR^2: {2:4f}'.format(slope, pval, rsq),
                transform=ax.transAxes, fontsize=12)

        # Save to PDF
        pdf.savefig(fig)
        plt.close(fig)

    # Make output table
    out = pd.Series({
        'slope': slope,
        'pval': pval,
        'flag_feature_runOrder_pval_05': flag_runOrder_pval_05,
        'flag_feature_runOrder_pval_01': flag_runOrder_pval_01,
        'rSquared': rsq
    })

    return out


def main(args):
    # Import data
    logger.info('Importing Data')
    dat = wideToDesign(args.fname, args.dname, args.uniqID, args.group, anno=[args.order, ])

    # Transpose Data so compounds are columns
    trans = dat.transpose()
    trans.set_index(args.order, inplace=True)

    # Drop group
    trans.drop(args.group, axis=1, inplace=True)

    # Run each column through regression
    logger.info('Running Regressions')

    res = list()
    # Iterate over columns
    # Was using an apply statment instead, but it kept giving me index errors
    for col in trans.columns:
        res.append(runOrder(trans[col]))

    # convert result list to a Series
    res = pd.Series({x[0]: x for x in res})

    # Drop rows that are missing regression results
    res.dropna(inplace=True)

    # Plot Results
    # Open a multiple page PDF for plots
    logger.info('Plotting Results')
    pp = PdfPages(args.oname)
    outTable = res.apply(makeScatter, args=(pp, ))

    # If there are no significant values, then output a page in the pdf
    # informing the user.
    if pp.get_pagecount() == 0:
        fig = plt.figure()
        fig.text(0.5, 0.4, 'There were no features significant for plotting.', fontsize=20)
        pp.savefig(fig)
    pp.close()

    # Write regression results to table
    outTable.index.name = args.uniqID
    outTable['flag_feature_runOrder_pval_01'] = outTable['flag_feature_runOrder_pval_01'].astype('int32')
    outTable['flag_feature_runOrder_pval_05'] = outTable['flag_feature_runOrder_pval_05'].astype('int32')
    outTable[['slope', 'pval', 'rSquared', 'flag_feature_runOrder_pval_01', 'flag_feature_runOrder_pval_05']].to_csv(args.tname, sep='\t', float_format='%.4f')


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
