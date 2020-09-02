#!/usr/bin/env python
######################################################################################
# Date: 2016/July/06
#
# Module: distribution_features.py
#
# VERSION: 1.0
#
# AUTHOR: Miguel Ibarra (miguelib@ufl.edu)
#
# DESCRIPTION: This program creates a distribution plot by features for a given datset
#######################################################################################
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
    """
    Function to pull in arguments
    """
    description = """ Distribution Summaries: The tool plots the distribution of the random subset of features. """
    parser = argparse.ArgumentParser(description=description, formatter_class=
                                    RawDescriptionHelpFormatter)
    # Standard Input
    standar = parser.add_argument_group(title='Standard input', 
                                description= 'Standard input for SECIM tools.')
    standar.add_argument("-i","--input", dest="input", action='store', 
                        required=True, help="Input dataset in wide format.")
    standar.add_argument("-d","--design", dest="design", action='store', 
                        required=True, help="Design file.")
    standar.add_argument("-id","--ID", dest="uniqID", action='store', 
                        required=True, help="Name of the column with uniqueID.")
    standar.add_argument("-g","--group", dest="group",default=False, action='store', 
                        required=False, help="Treatment group")
    # Tool output
    output = parser.add_argument_group(title="Output paths", 
                                        description="Paths and outputs")
    output.add_argument("-f","--figure", dest="figure", action='store',
                        required=True, help="Output figure name [pdf].")
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

    return(args)


def plotDensity (data, name, pdf):
    """
    This function takes a pandas dataframe and plots a
    density plot and a boxplot.
    """
    # Stablishing figure layout (x,y,colspan,rowspan)
    axisLayout = [(0,0,1,3),(3,0,1,1)]

    # Creating a figure template
    figure = figureHandler(proj='2d', numAx=2, numRow=4, numCol=1, 
                            figsize=(8,13), arrangement=axisLayout)
    # Adding figure Title
    figure.formatAxis(figTitle="Distribution by Features {0}".format(name),
                    xlim="ignore", ylim="ignore",axnum=0,showX=False)

    #Creting list of len(wide.T) maximu 50 with the colors for each index
    colors =  [palette.ugColors[name]] * len(data.index)

    # Plotting boxPlot
    box.boxDF(ax=figure.ax[0], colors=colors, dat=data.T,
             vert=False,rot=0)

    # Plotting density plot
    density.plotDensityDF(data=data.T.unstack(), 
                    ax=figure.ax[1], colors=colors[0])

    # Adding figure to pdf object
    figure.addToPdf(pdf)


def main(args):
    """
    Function to call all other functions
    """
    # Loading files with interface
    logger.info("Loading data with the Interface")
    dat = wideToDesign(args.input,args.design,args.uniqID,group=args.group, logger=logger)

    # Cleaning from missing data
    dat.dropMissing()

    # Subseting wide to get features for wide files with more that 50 features
    if len(dat.wide.index) > 50:
        wide = dat.wide.sample(n=50,axis=0)
        wide = wide.T
    else:
        wide = dat.wide.T

    # Saving figure
    with PdfPages(args.figure) as pdf:
        # Iterating over groups
        if args.group:

            # Getting colors for groups
            palette.getColors(design=dat.design, groups=[dat.group])

            # Iterating over groups
            for name, group in dat.design.groupby(args.group):
                logger.info("Plotting for group {0}".format(name))

                # Plotting Density and Box plot for the group
                plotDensity(data=wide.T[group.index],name=name,pdf=pdf)

        # Get colors for each feature for "All groups"
        logger.info("Plotting for group {0}".format("samples"))
        palette.getColors(design=dat.design, groups=[])

        # Plotting density and boxplots for all
        plotDensity(data=wide, name="samples", pdf=pdf)

        #Ending script
        logger.info("Ending script")


if __name__ == '__main__':
    args = getOptions()
    logger = logging.getLogger()
    sl.setLogger(logger)
    logger.info("Importing data with following parameters: "\
            "\n\tWide: {0}"\
            "\n\tDesign: {1}"\
            "\n\tUnique ID: {2}".format(args.input, args.design, args.uniqID))
    palette = colorHandler(pal=args.palette, col=args.color)
    logger.info("Using {0} color scheme from {1} palette".format(args.color,
                args.palette))
    main(args)
