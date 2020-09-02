#!/usr/bin/env python
################################################################################
# Date: 2016/July/06 ed. 1016/July/11
# 
# Module: distribution_samples.py
#
# VERSION: 1.1
# 
# AUTHOR: Miguel Ibarra (miguelib@ufl.edu) 
#         Edited by Matt Thoburn (mthoburn@ufl.edu)
#
# DESCRIPTION: This program creates a distribution plot by samples for a given 
#              datset
#
################################################################################
# Import built-in libraries
import os
import logging
import argparse
from argparse import RawDescriptionHelpFormatter

# Import add-on libraries
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Import local data libraries
from secimtools.dataManager import logger as sl
from secimtools.dataManager.interface import wideToDesign

# Import local plotting libraries
from secimtools.visualManager import module_box as box
from secimtools.visualManager import module_distribution as density
from secimtools.visualManager.manager_color import colorHandler
from secimtools.visualManager.manager_figure import figureHandler

def getOptions():
    """ Function to pull in arguments """
    description = """ Distribution Summaries: The tool plots the distribution of samples. """
    parser = argparse.ArgumentParser(description=description, 
                                    formatter_class=RawDescriptionHelpFormatter)
    standard = parser.add_argument_group(title="Standar input", 
        description="Standar input for SECIM tools.")
    # Standard Input
    standard.add_argument("-i","--input", dest="input", action='store', 
                        required=True, help="Input dataset in wide format.")
    standard.add_argument("-d","--design", dest="design", action='store', 
                        required=True, help="Design file.")
    standard.add_argument("-id","--ID", dest="uniqID", action='store', 
                        required=True, help="Name of the column with uniqueID.")
    standard.add_argument("-g","--group", dest="group", action='store', 
                        required=False, default=False, help="Name of column in "\
                        "design file with Group/treatment information.")
    standard.add_argument("-o",'--order', dest='order', action='store', 
                        required=False, default=False, help="Name of the column "\
                        "with the runOrder")
    standard.add_argument("-l","--levels",dest="levels",action="store", 
                        required=False, default=False, help="Different groups to"\
                         "sort by separeted by commas.")
    # Tool Output
    output = parser.add_argument_group(title="Output paths", description="""Paths 
                                        and outputs""")
    output.add_argument("-f","--figure",dest="figure",action="store",required=True,
                        help="Path for the distribution figure")
    # Plot options
    plot = parser.add_argument_group(title='Plot options')
    plot.add_argument("-pal","--palette",dest="palette",action='store',required=False, 
                        default="tableau", help="Name of the palette to use.")
    plot.add_argument("-col","--color",dest="color",action="store",required=False, 
                        default="Tableau_20", help="Name of a valid color scheme"\
                        " on the selected palette")
    args = parser.parse_args()

    # Standardize paths
    args.input  = os.path.abspath(args.input)
    args.design = os.path.abspath(args.design)
    args.figure = os.path.abspath(args.figure)

    # if args.levels then split otherwise args.level = emptylist
    if args.levels:
        args.levels = args.levels.split(",")
    return(args)

def plotDensityDistribution(pdf,wide,palette):
    # Instanciating figureHandler object
    figure = figureHandler(proj="2d", figsize=(12,7))

    # Formating axis
    figure.formatAxis(figTitle="Distribution by Samples Density",xlim="ignore",
                        ylim="ignore",grid=False)

    # Plotting density plot
    density.plotDensityDF(colors=palette.design["colors"],ax=figure.ax[0],data=wide)

    # Add legend to the plot
    figure.makeLegend(ax=figure.ax[0], ucGroups=palette.ugColors, group=palette.combName)

    # Shrinking figure
    figure.shrink()

    # Adding to PDF
    figure.addToPdf(pdf, dpi=600)

def plotBoxplotDistribution(pdf,wide,palette):
    # Instanciating figureHandler object
    figure = figureHandler(proj="2d",figsize=(max(len(wide.columns)/4,12),7))

    # Formating axis
    figure.formatAxis(figTitle="Distribution by Samples Boxplot",ylim="ignore",
                    grid=False,xlim="ignore")

    # Plotting boxplot
    box.boxDF(ax=figure.ax[0],colors=palette.design["colors"],dat=wide)

    # Shrinking figure
    figure.shrink()

    #Adding to PDF
    figure.addToPdf(pdf, dpi=600)

def main(args):
    """
    Function to call all other functions
    """
    # Checking if levels
    if args.levels and args.group:
        levels = [args.group]+args.levels
    elif args.group and not args.levels:
        levels = [args.group]
    else:
        levels = []
    logger.info("Groups used to color by: {0}".format(",".join(levels)))

    # Parsing files with interface
    logger.info("Loading data with the Interface")
    dat = wideToDesign(args.input, args.design, args.uniqID, args.group,
                    anno=args.levels, runOrder=args.order, logger=logger)

    # Cleaning from missing data
    dat.dropMissing()
    
    # Sort data by runOrder if provided
    if args.order:
       logger.info("Sorting by runOrder")
       design_final = dat.design.sort_values(by=args.order, axis=0)
       wide_final = dat.wide.reindex(columns= design_final.index )
		
    else:
       design_final = dat.design
       wide_final = dat.wide


    # Get colors for each sample based on the group
    palette.getColors(design=design_final,groups=levels)

    # Open PDF pages to output figures
    with PdfPages(args.figure) as pdf:

        # Plot density plot
        logger.info("Plotting density for sample distribution")
        plotDensityDistribution(pdf=pdf, wide=wide_final, palette=palette)

        # Plot boxplots
        logger.info("Plotting boxplot for sample distribution")
        plotBoxplotDistribution(pdf=pdf, wide=wide_final, palette=palette)
        
    logger.info("Script complete!")

if __name__ == '__main__':
    # import Arguments
    args = getOptions()

    # Setting logging
    logger = logging.getLogger()
    sl.setLogger(logger)

    # Print logger info
    logger.info("""Importing data with following parameters: 
            \tWide: {0}
            \tDesign: {1}
            \tUnique ID: {2}
            \tGroup: {3}
            \tRun Order: {4}
            \tLevels: {5}
            """ .format(args.input, args.design, args.uniqID, args.group, 
                args.order,args.levels))

    # Set color palette
    palette = colorHandler(pal=args.palette, col=args.color)
    logger.info("Using {0} color scheme from {1} palette".format(args.color,
                args.palette))
    # main
    main(args)
