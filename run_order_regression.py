#!/usr/bin/env python
################################################################################
# DATE: 2017/01/04
#
# SCRIPT: runOrderRegression.py
#
# VERSION: 1.2
# 
# AUTHOR: Miguel A Ibarra (miguelib@ufl.edu) 
#           Edited by: Matt Thoburn (mthoburn@ufl.edu)
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
import statsmodels.formula.api as smf
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# Local Packages
import logger as sl
from flags import Flags
from interface import wideToDesign
from generalModules.drop_missing import dropMissing

# Local plotting modules
import module_lines as lines
import module_scatter as scatter
from manager_color import colorHandler
from manager_figure import figureHandler

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
    group2.add_argument("-fl","--flags", dest="flags", action="store", 
                        required=True, help="Name of table flags")

    group4 = parser.add_argument_group(title="Development Settings")
    group4.add_argument("-dg","--debug", dest="debug", action="store_true", 
                        required=False, help="Add debugging log output.")

    plot = parser.add_argument_group(title='Plot options')
    plot.add_argument("-pal","--palette",dest="palette",action='store',
                    required=False, default="tableau", help="Name of the "\
                    "palette to use.")
    plot.add_argument("-col","--color",dest="color",action="store",
                    required=False, default="Tableau_20", help="Name of a valid"\
                    " color scheme on the selected palette")


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
        slope = results.params["run"]
        pval = results.pvalues["run"]
        rsq = results.rsquared
        fitted = results.fittedvalues

        #Returning
        out =  pd.Series(name=col.name,data={"name":col.name,
                "x":clean["run"], 
                "y":clean["val"], 
                "res":results, 
                "slope":slope, 
                "pval":pval,
                "rsq":rsq,
                "fitted":fitted})

        return out

    else:
        logger.warn("""{} had fewer than 5 samples without NaNs, it will be 
                    ignored.""".format(col.name))

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
    # Get 95% CI
    prstd, lower, upper = wls_prediction_std(row["res"])

    # Sort CIs for Plotting
    toPlot = pd.DataFrame({"x": row["x"], "lower": lower, "upper": upper})
    toPlot.sort_values(by="x", inplace=True)

    # Get colors
    des, ucolor, combname = palette.getColors(des,[groupName])
    colors, ucolors = palette.getColorsByGroup(des,groupName, ucolor)

    # Create plot
    fh = figureHandler(proj="2d", figsize=(14,8))
    
    #Plot scatterplot
    scatter.scatter2D(ax=fh.ax[0],x=row["x"],y=row["y"],colorList=colors)

    # Plot cutoffs
    lines.drawCutoff(ax=fh.ax[0],x=toPlot["x"],y=toPlot["lower"],c="r")
    lines.drawCutoff(ax=fh.ax[0],x=row["x"],y=row["fitted"],c="c")
    lines.drawCutoff(ax=fh.ax[0],x=toPlot["x"],y=toPlot["upper"],c="r")

    # Formatting
    ymin, ymax = fh.ax[0].get_ylim()
    fh.formatAxis(xTitle="Run Order", yTitle="Value", ylim=(ymin,ymax*1.2),
    figTitle=u"{} Scatter plot (fitted regression line and prediction bands"\
    " included)".format(row["name"]))

    # Shrink figure
    fh.shrink()

    # Add legend to figure
    fh.makeLegend(ax=fh.ax[0], ucGroups=ucolor, group=groupName)

    #Add text to the ax
    fh.ax[0].text(.7, .85, u"Slope= {0:.4f}\n(p-value = {1:.4f})\n"\
        "$R^2$ = {2:4f}".format(round(row["slope"],4), round(row["pval"],4), 
        round(row["rsq"],4)),transform=fh.ax[0].transAxes, fontsize=12)

    # Save to PDF
    fh.addToPdf(pdf)

def main(args):
    # Import data
    logger.info("Importing Data")

    #Parsing data with interface
    dat = wideToDesign(args.input, args.design, args.uniqID, args.group, 
                        anno=[args.order, ])

    # Treat everything as numeric
    dat.wide = dat.wide.applymap(float)

    #Dropping missing values
    if np.isnan(dat.wide.values).any():
        dat.wide = dropMissing(dat.wide,logger=logger)

    # Transpose Data so compounds are columns
    trans = dat.transpose()
    trans.set_index(args.order, inplace=True)

    # Drop group
    trans.drop(args.group, axis=1, inplace=True)

    # Run each column through regression
    logger.info("Running Regressions")

    # Was using an apply statment instead, but it kept giving me index errors
    res = [runOrder(trans[col]) for col in trans.columns if runOrder(trans[col]) is not None]

    # Concatenation results into a single dataframe
    res_df = pd.concat(res,axis=1)
    
    # Transpose data
    res_df = res_df.T

    # Drop rows that are missing regression results
    res_df.dropna(inplace=True)

    # Getting table out    
    out_df=res_df[["pval","rsq","slope"]]

    # Plot Results
    # Open a multiple page PDF for plots
    isscatter = False
    logger.info("Plotting Results")
    with PdfPages(args.figure) as pdf:
        # Iterates over all rows in the dataframe
        for index,row in res_df.iterrows():
            # Make scatter plot if p-pvalue is less than 0.05
            if row["pval"] <= 0.05:
                makeScatter(row,pdf,dat.design,args.group)
                isscatter = True
        # If not plot was generated then output an empty pdf with a message.
        if not(isscatter):
            fig = plt.figure()
            fig.text(0.5, 0.4, "There were no features significant for plotting.", 
            fontsize=12)
            pdf.savefig(fig)

    # Setting flag object
    ror_flags = Flags(index=out_df.index)
    
    # Adding flags for pval 0.05 and 0.1
    ror_flags.addColumn(column="flag_feature_runOrder_pval_05",
                        mask=(out_df["pval"]<=0.05))
    ror_flags.addColumn(column="flag_feature_runOrder_pval_01",
                        mask=(out_df["pval"]<=0.1))

    # Write regression results to table
    out_df.to_csv(args.table, sep="\t", float_format="%.4f", 
                    index_label=args.uniqID)

    # Write regression flags to table
    ror_flags.df_flags.to_csv(args.flags, sep="\t", index_label=args.uniqID)

if __name__ == "__main__":
    # Command line options
    args = getOptions()

    # Setting up logger
    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel="debug")
        DEBUG = True
    else:
        sl.setLogger(logger)

    # Set color palette
    palette = colorHandler(pal=args.palette, col=args.color)
    logger.info(u"Using {0} color scheme from {1} palette".format(args.color,
                args.palette))
    # Runnign code
    main(args)
