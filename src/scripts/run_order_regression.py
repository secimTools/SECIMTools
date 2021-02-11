#!/usr/bin/env python
################################################################################
# AUTHORS: Miguel A Ibarra <miguelib@ufl.edu>
#          Matt Thoburn <mthoburn@ufl.edu>
#
# DESCRIPTION: Take a a wide format file and makes a run order regression on it.
################################################################################

from __future__ import division
import os
import sys
import logging
import argparse
from argparse import RawDescriptionHelpFormatter
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from secimtools.dataManager import logger as sl
from secimtools.dataManager.flags import Flags
from secimtools.dataManager.interface import wideToDesign
from secimtools.visualManager import module_lines as lines
from secimtools.visualManager import module_scatter as scatter
from secimtools.visualManager.manager_color import colorHandler
from secimtools.visualManager.manager_figure import figureHandler


def getOptions():
    """ Function to pull in arguments """
    description = """ """
    parser = argparse.ArgumentParser(description=description,
                                    formatter_class=RawDescriptionHelpFormatter)
    # Standard Input
    standard = parser.add_argument_group(title="Standard input",
                        description="Standard input for SECIM tools.")
    standard.add_argument("-i","--input", dest="input", action="store",
                        required=True, help="Input dataset in wide format.")
    standard.add_argument("-d","--design", dest="design", action="store",
                        required=True, help="Design file.")
    standard.add_argument("-id","--ID", dest="uniqID", action="store", required=True,
                        help="Name of the column with unique identifiers.")
    standard.add_argument("-g","--group", dest="group", action="store", required=True,
                        help="Group/treatment identifier in design file [Optional].")
    standard.add_argument("-o","--order", dest="order", action="store", required=True,
                        help="Name of the column on design file that contains "\
                        "run order")
    standard.add_argument("-l","--levels",dest="levels",action="store",
                        required=False, default=False, help="Different groups to"\
                        " sort by separeted by commas.")
    # Tool Output
    output = parser.add_argument_group(title="Required Output")
    output.add_argument("-f","--fig", dest="figure", action="store",
                        required=True, help="Name of PDF to save scatter plots.")
    output.add_argument("-t","--table", dest="table", action="store",
                        required=True, help="Name of table for scatter plots")
    output.add_argument("-fl","--flags", dest="flags", action="store",
                        required=True, help="Name of table flags")
    # Dev Options
    dev = parser.add_argument_group(title="Development Settings")
    dev.add_argument("-dg","--debug", dest="debug", action="store_true",
                        required=False, help="Add debugging log output.")
    # Plot Options
    plot = parser.add_argument_group(title='Plot options')
    plot.add_argument("-pal","--palette",dest="palette",action='store',
                    required=False, default="tableau", help="Name of the "\
                    "palette to use.")
    plot.add_argument("-col","--color",dest="color",action="store",
                    required=False, default="Tableau_20", help="Name of a valid"\
                    " color scheme on the selected palette")
    args = parser.parse_args()
    # Standardize paths
    args.input  = os.path.abspath(args.input)
    args.table  = os.path.abspath(args.table)
    args.flags  = os.path.abspath(args.flags)
    args.figure = os.path.abspath(args.figure)
    args.design = os.path.abspath(args.design)
    if args.levels:
        args.levels = args.levels.split(",")
    return (args)


def runRegression(data):
    """
    Run each column through regression and then concatenate the outputed series
    to a single dataframe.

    :Arguments:
        :type data: pandas.dataFrame
        :param data: A Pandas dataFrame with a transpose version of the original
                    data and runOrder as index.

    :Returns:
        :rtype res_df: pandas.dataFrame
        :return res_df: Results of the regression.
    """
    res = list()
    for colName in data.columns:
        column = data[colName]
        # Drop missing values and make sure everything is numeric
        clean = column.dropna().reset_index()
        clean.columns = ["run", "val"]
        try:
            clean["run"] = clean["run"].astype('float64')
        except ValueError as e:
            logger.error("Invalid runOrder value found: - {}".format(str(e)))
            sys.exit(1)

        # Fit model
        model   = smf.ols(formula="val ~ run", data=clean, missing="drop")
        results = model.fit()
        slope   = results.params["run"]
        pval    = results.pvalues["run"]
        rsq     = results.rsquared
        fitted  = results.fittedvalues

        #Returning
        out =  pd.Series(name=column.name,
                data={"name":column.name,
                "x":clean["run"],
                "y":clean["val"],
                "res":results,
                "slope":slope,
                "pval":pval,
                "rsq":rsq,
                "fitted":fitted})
        res.append(out)

    # Concatenating results
    res_df = pd.concat(res,axis=1)
    res_df = res_df.T

    # Drop rows that are missing regression results
    res_df.dropna(inplace=True)

    # Returning results of regressions
    return (res_df)


def plotSignificantROR(data, pdf, palette):
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
    # Iterates over all rows in the dataframe
    # Make scatter plot if p-pvalue is less than 0.05
    for index,row in data.iterrows():
        if row["pval"] > 0.05: continue
        #plotSignificantROR(row,pdf,dat.design,args.group)

        # Get 95% CI
        prstd, lower, upper = wls_prediction_std(row["res"])

        # Sort CIs for Plotting
        toPlot = pd.DataFrame({"x": row["x"], "lower": lower, "upper": upper})
        toPlot.sort_values(by="x", inplace=True)

        # Create plot
        fh = figureHandler(proj="2d", figsize=(14,8))

        #Plot scatterplot
        scatter.scatter2D(ax=fh.ax[0],x=row["x"],  y=row["y"],colorList=palette.list_colors)

        # Plot cutoffs
        lines.drawCutoff(ax=fh.ax[0],x=row["x"],   y=row["fitted"],  c="c")
        lines.drawCutoff(ax=fh.ax[0],x=toPlot["x"],y=toPlot["lower"],c="r")
        lines.drawCutoff(ax=fh.ax[0],x=toPlot["x"],y=toPlot["upper"],c="r")

        # Formatting
        ymin, ymax = fh.ax[0].get_ylim()
        fh.formatAxis(xTitle="Run Order", yTitle="Value", ylim=(ymin,ymax*1.2),
        figTitle=u"{} Scatter plot (fitted regression line and prediction bands"\
        " included)".format(row["name"]))

        # Shrink figure
        fh.shrink()

        # Add legend to figure
        fh.makeLegend(ax=fh.ax[0], ucGroups=palette.ugColors, group=palette.combName)

        #Add text to the ax
        fh.ax[0].text(.7, .85, u"Slope= {0:.4f}\n(p-value = {1:.4f})\n"\
            "$R^2$ = {2:4f}".format(round(row["slope"],4), round(row["pval"],4),
            round(row["rsq"],4)),transform=fh.ax[0].transAxes, fontsize=12)

        # Save to PDF
        fh.addToPdf(pdf)


def main(args):
    """Perform run order regression analysis"""

    if args.levels and args.group:
        levels = [args.group]+args.levels
    elif args.group and not args.levels:
        levels = [args.group]
    else:
        levels = []

    logger.info("Loading data with the Interface")
    dat = wideToDesign(args.input, args.design, args.uniqID, args.group,
                        runOrder=args.order, anno=args.levels, logger=logger)

    # Remove samples with missing data
    dat.dropMissing()
    # Get remaining Sample IDs for dataframe filtering
    sample_ids = dat.wide.index.tolist()
    # Set color palette and get colors
    palette = colorHandler(pal=args.palette, col=args.color)
    logger.info(u"Using {0} color scheme from {1} palette".format(args.color, args.palette))
    palette.getColors(dat.design,levels)

    # Transpose Data so compounds are columns, set the runOrder as index
    # and drop the colum with the groups from the tranposed wide.
    trans = dat.transpose()
    trans.set_index(dat.runOrder, inplace=True)
    trans.drop(dat.group, axis=1, inplace=True)
    trans = trans.loc[:,sample_ids]
    # Run regressions
    logger.info("Running Regressions")
    ror_df = runRegression(trans)

    # Creating flags flags for pvals 0.05 and 0.1
    ror_flags = Flags(index=ror_df.index)
    ror_flags.addColumn(column="flag_feature_runOrder_pval_05",
                        mask=(ror_df["pval"]<=0.05))
    ror_flags.addColumn(column="flag_feature_runOrder_pval_01",
                        mask=(ror_df["pval"]<=0.01))

    # Plot Results
    # Open a multiple page PDF for plots
    logger.info("Plotting Results")
    with PdfPages(args.figure) as pdf:
        plotSignificantROR(ror_df, pdf, palette)

        # If not pages
        if pdf.get_pagecount() == 0:
            fig = plt.figure()
            fig.text(0.5, 0.4, "There were no features significant for plotting.", fontsize=12)
            pdf.savefig(fig)

    # Write  results and flasg to TSV files
    ror_df.to_csv(args.table, sep="\t", float_format="%.4f", index_label=args.uniqID,
                columns=["pval","rsq","slope"])
    ror_flags.df_flags.to_csv(args.flags, sep="\t", index_label=args.uniqID)


if __name__ == "__main__":
    """Run if executed on the command line"""
    args = getOptions()
    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel="debug")
        DEBUG = True
    else:
        sl.setLogger(logger)
    # Print logger info
    logger.info(u"""Importing data with following parameters:
            \tWide: {0}
            \tDesign: {1}
            \tUnique ID: {2}
            \tGroup: {3}
            \tRun Order: {4}
            \tLevels: {5}
            """ .format(args.input, args.design, args.uniqID, args.group,
                args.order,args.levels))

    main(args)

