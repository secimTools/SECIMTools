#!/usr/bin/env python
################################################################################
# DATE: 2016/July/11
#
# SCRIPT: runOrderRegression.py
#
# VERSION: 1.1
# 
# AUTHOR: Miguel A Ibarra (miguelib@ufl.edu) Edited by: Matt Thoburn (mthoburn@ufl.edu)
# 
# DESCRIPTION: This script takes a a wide format file and makes a run 
#   order regression on it.
#
################################################################################
# Built-in packages
from __future__ import division
import logging
import argparse
from argparse import RawDescriptionHelpFormatter

# Add-on packages
import numpy as np

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# Local Packages
from interface import wideToDesign
from flags import Flags
import logger as sl

from manager_color import colorHandler
from manager_figure import figureHandler

import module_lines as lines
import module_scatter as scatter

def getOptions():
    """ Function to pull in arguments """
    description = """ """
    parser = argparse.ArgumentParser(description=description, 
                                    formatter_class=RawDescriptionHelpFormatter)
    group1 = parser.add_argument_group(title="Standard input", 
                                    description="Standard input for SECIM tools.")
    group1.add_argument("-i","--input", dest="input", action="store", 
                        required=True, help="Input dataset in wide format.")
    group1.add_argument("-d","--design", dest="design", action="store",
                        required=True, help="Design file.")
    group1.add_argument("-id","--ID", dest="uniqID", action="store", required=True,
                        help="Name of the column with unique identifiers.")
    group1.add_argument("-g","--group", dest="group", action="store", required=True,
                        help="Group/treatment identifier in design file [Optional].")
    group1.add_argument("-o","--order", dest="order", action="store", required=True, 
                        help="""Name of the column on design file that contauns 
                        run order""")

    group2 = parser.add_argument_group(title="Required Output")
    group2.add_argument("-f","--fig", dest="figure", action="store", 
                        required=True, help="Name of PDF to save scatter plots.")
    group2.add_argument("-t","--table", dest="table", action="store", 
                        required=True, help="Name of table for scatter plots")

    group4 = parser.add_argument_group(title="Development Settings")
    group4.add_argument("-dg","--debug", dest="debug", action="store_true", 
                        required=False, help="Add debugging log output.")

    args = parser.parse_args()

    return (args)

def runOrder(col):
    """ 
    Run order regression

    :Arguments:
        :type col: pandas.Series
        :param col: A Pandas series with run order ["run"] as the index
            and ["val"] as the value to regressed on..

    :Returns:
        :rtype name: str
        :return name: Name of the column.

        :rtype X: int
        :return X: Values of X ["run"]

        :rtype Y: int
        :return Y: Values of Y ["val"]

        :rtype results: sm.RegressionResults
        :return results: Regression results    
    """

    # Drop missing values and make sure everything is numeric
    clean = col.dropna().reset_index()
    clean.columns = ["run", "val"]

    if clean.shape[0] >= 5:
        # Fit model
        model = smf.ols(formula="val ~ run", data=clean, missing="drop")
        results = model.fit()

        #Returning
        return col.name, clean["run"], clean["val"], results
    else:
        logger.warn("""{} had fewer than 5 samples without NaNs, it will be 
                    ignored.""".format(col.name))

def flagPval(val, alpha):
    if val <= alpha:
        flag = True
    else:
        flag = False

    return flag

def makeScatter(row, pdf,des,groupName):
    """ 
    Plot a scatter plot of x vs y. 

    :Arguments:

        :type row:
        :param row:

        :type pdf: PdfPages
        :param pdf: pdf object to store scatterplots

        :type des: pandas DataFrame
        :param des: design file

        :type groupName: string
        :param groupName: name of group
    """
    # Split out row results
    name, x, y, res = row

    # Get fitted values
    fitted = res.fittedvalues

    # Get slope and p-value
    slope = res.params["run"]
    pval = res.pvalues["run"]
    rsq = res.rsquared

    # Make Flag based on p-value
    flag_runOrder_pval_05 = flagPval(pval, 0.05)
    flag_runOrder_pval_01 = flagPval(pval, 0.01)

    #Get colors
    ch = colorHandler("tableau","Tableau_20")

    #print groups
    des,ucolor,combname=ch.getColors(des,[groupName])
    colors,ucolors = ch.getColorsByGroup(des,groupName,ucolor)

    # Make scatter plot if p-pvalue is less than 0.05
    if flag_runOrder_pval_05 == 1:
        # Get 95% CI
        prstd, lower, upper = wls_prediction_std(res)

        # Sort CIs for Plotting
        toPlot = pd.DataFrame({"x": x, "lower": lower, "upper": upper})
        toPlot.sort_values(by="x", inplace=True)

        # Plot
        fh = figureHandler(proj="2d")

        #Plot scatterplot
        scatter.scatter2D(ax=fh.ax[0],x=x,y=y,colorList=colors)

        # Plot cutoffs
        lines.drawCutoff(ax=fh.ax[0],x=toPlot["x"],y=toPlot["lower"],c="r")
        lines.drawCutoff(ax=fh.ax[0],x=x,y=fitted,c="c")
        lines.drawCutoff(ax=fh.ax[0],x=toPlot["x"],y=toPlot["upper"],c="r")

        #formatting
        fh.formatAxis(xTitle="Run Order", yTitle="Value", 
            figTitle=u"""{} Scatter plot
             (fitted regression line and prediction bands included)""".format(name))

        #Shrink figure to add legend
        fh.shrink()

        #Add legend to figure
        fh.makeLegend(ax=fh.ax[0],ucGroups=ucolor,group=groupName)

        #Add text to the ax
        fh.ax[0].text(.7, .85, u"Slope= {0:.4f}\n(p-value = {1:.4f})\n$R^2$ = {2:4f}".format(
            round(slope,4), round(pval,4), round(rsq,4)),transform=fh.ax[0].transAxes, fontsize=12)

        # Save to PDF
        fh.addToPdf(pdf)

    # Make output table
    out = pd.Series({"slope": slope,"pval": pval,"rSquared": rsq,
        "flag_feature_runOrder_pval_05": flag_runOrder_pval_05,
        "flag_feature_runOrder_pval_01": flag_runOrder_pval_01})

    return out

def main(args):
    # Import data
    logger.info("Importing Data")

    #Parsing data with interface
    dat = wideToDesign(args.input, args.design, args.uniqID, args.group, 
                        anno=[args.order, ])

    # Transpose Data so compounds are columns
    trans = dat.transpose()
    trans.set_index(args.order, inplace=True)

    # Drop group
    trans.drop(args.group, axis=1, inplace=True)

    # Run each column through regression
    logger.info("Running Regressions")

    # Was using an apply statment instead, but it kept giving me index errors
    res = [runOrder(trans[col]) for col in trans.columns if runOrder(trans[col]) is not None]

    # convert result list to a Series
    res = pd.Series({x[0]: x for x in res})

    # Drop rows that are missing regression results
    res.dropna(inplace=True)

    # Plot Results
    # Open a multiple page PDF for plots
    logger.info("Plotting Results")
    pdf = PdfPages(args.figure)
    outTable = res.apply(makeScatter, args=(pdf,dat.design,args.group))

    # If there are no significant values, then output a page in the pdf
    # informing the user.
    if pdf.get_pagecount() == 0:
        fig = plt.figure()
        fig.text(0.5, 0.4, "There were no features significant for plotting.", fontsize=12)
        pdf.savefig(fig)
    pdf.close()

    # Write regression results to table
    outTable.index.name = args.uniqID
    outTable["flag_feature_runOrder_pval_01"] = outTable["flag_feature_runOrder_pval_01"].astype("int32")
    outTable["flag_feature_runOrder_pval_05"] = outTable["flag_feature_runOrder_pval_05"].astype("int32")
    outTable[["slope", "pval", "rSquared", "flag_feature_runOrder_pval_01", "flag_feature_runOrder_pval_05"]].to_csv(args.table, sep="\t", float_format="%.4f")

if __name__ == "__main__":
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel="debug")
        DEBUG = True
    else:
        sl.setLogger(logger)

    main(args)
