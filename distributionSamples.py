#!/usr/bin/env python
######################################################################################
# Date: 2016/July/06 ed. 1016/July/11
# 
# Module: distribution_samples.py
#
# VERSION: 1.1
# 
# AUTHOR: Miguel Ibarra (miguelib@ufl.edu) Edited by Matt Thoburn (mthoburn@ufl.edu)
#
# DESCRIPTION: This program creates a distribution plot by samples for a given datset
#
#######################################################################################

# Built-in packages
import os
import logging
import argparse
from argparse import RawDescriptionHelpFormatter

# Add-on packages
import mpld3
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Local Packages
import logger as sl
import module_box as box
import module_distribution as density
from interface import wideToDesign
from manager_color import colorHandler
from manager_figure import figureHandler

def getOptions():
    """ Function to pull in arguments """
    description = """ Distribution Analysis: Plot sample distrubtions. """
    parser = argparse.ArgumentParser(description=description, 
                                    formatter_class=RawDescriptionHelpFormatter)
    standar = parser.add_argument_group(title="Standar input", description="""
                                        Standar input for SECIM tools.""")
    standar.add_argument("-i","--input", dest="fname", action='store', 
                        required=True, help="Input dataset in wide format.")
    standar.add_argument("-d","--design", dest="dname", action='store', 
                        required=True, help="Design file.")
    standar.add_argument("-id","--ID", dest="uniqID", action='store', 
                        required=True, help="Name of the column with uniqueID.")

    output = parser.add_argument_group(title="Output paths", description="""Paths 
                                        and outputs""")
    output.add_argument("-f","--figure",dest="figure",action="store",required=True,
                        help="Path for the distribution figure")

    optional = parser.add_argument_group(title="Optional input", 
                        description="Optional input for districution script.")
    optional.add_argument("-g","--group", dest="group", action='store', 
                        required=False, default=False, help="""Name of column in 
                        design file with Group/treatment information.""")
    optional.add_argument("-o",'--order', dest='order', action='store', 
                        required=False, default=False, help="""Name of the column 
                        with the runOrder""")
    optional.add_argument("-l","--levels",dest="levels",action="store", 
                        required=False, default=False, help="""Different groups to
                         sort by separeted by commas.""")

    args = parser.parse_args()
    return(args)

def plotDensityDistribution(pdf,wide,design,uGroups,lvlName):
    # Instanciating figureHandler object
    logger.info(u"Plotting density for sample distribution")
    figure = figureHandler(proj="2d", figsize=(12,7))

    # Formating axis
    figure.formatAxis(figTitle="Distribution by Samples Density",xlim="ignore",
                        ylim="ignore",grid=False)

    # Plotting density plot
    density.plotDensityDF(colors=design["colors"],ax=figure.ax[0],data=wide)

    # Add legend to the plot
    figure.makeLegend(ax=figure.ax[0], ucGroups=uGroups, group=lvlName)

    # Shrinking figure
    figure.shrink()

    # Adding to PDF
    figure.addToPdf(pdf, dpi=600)

def plotBoxplotDistribution(pdf,wide,design):
    # Instanciating figureHandler object
    logger.info(u"Plotting boxplot for sample distribution")
    figure = figureHandler(proj="2d",figsize=(max(len(wide.columns)/4,12),7))

    # Formating axis
    figure.formatAxis(figTitle="Distribution by Samples Boxplot",ylim="ignore",
                    grid=False,xlim="ignore")

    # Plotting boxplot
    box.boxDF(ax=figure.ax[0],colors=design["colors"],dat=wide)

    # Shrinking figure
    plt.tight_layout()

    #Adding to PDF
    figure.addToPdf(pdf, dpi=600)

def main(args):
    """
    Function to call all other functions
    """
    logger.info(u"""Importing data with following parameters: 
                \tWide: {0}
                \tDesign: {1}
                \tUnique ID: {2}
                \tGroup: {3}
                \tRun Order: {4}
                \tLevels: {5}
                """ .format(args.fname, args.dname, args.uniqID, args.group, 
                    args.order,args.levels))

    # Checking if levels
    subGroups = []
    if args.levels and args.group:
        subGroups = args.levels.split(",")
        levels = [args.group]+subGroups
    elif args.group and not args.levels:
        levels = [args.group]
    elif not args.group and args.levels:
        logger.error(u"Use --groups if you are using one group to select")
        exit()
    else:
        levels = []
    logger.info(u"Groups used to color by :{}".format(",".join(levels)))

    # Checkig if order
    if args.order and args.levels:
        anno = [args.order] + subGroups
    elif args.order and not args.levels:
        anno = [args.order,]
    elif not args.order and args.levels:
        anno = [args.levels,]
    else:
        anno=False

    # Parsing files with interface
    dat = wideToDesign(args.fname,args.dname,args.uniqID,args.group,anno=anno)
    logger.info(u"Data loaded by interface")
    
    # Sort data if runOrder
    if args.order:
        dat.design.sort_values(by=args.order,inplace=True)
        dat.wide = dat.wide[dat.design.index.values]
        logger.info(u"Sorting by runOrderf")

    # Set palette of colors
    palette = colorHandler(pal="tableau", col="Tableau_20")
    logger.info(u"Using tableau Tableau_20 palette")

    # Get colors for each sample based on the group
    dat.design,uGroups,combName=palette.getColors(design=dat.design,groups=levels)

    # Open PDF pages to output figures
    with PdfPages(os.path.abspath(args.figure)) as pdf:
        # Plot density plot
        plotDensityDistribution(pdf=pdf,wide=dat.wide,design=dat.design,
                                uGroups=uGroups, lvlName=combName)

        # Plot boxplots
        plotBoxplotDistribution(pdf=pdf,wide=dat.wide,design=dat.design)
        
    logger.info(u"Ending script")


if __name__ == '__main__':
    # import Arguments
    args = getOptions()

    # Setting logging
    logger = logging.getLogger()
    sl.setLogger(logger)

    # main
    main(args)
