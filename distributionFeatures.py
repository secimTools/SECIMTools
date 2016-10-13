######################################################################################
# Date: 2016/July/06
# 
# Module: distribution_feature.py
#
# VERSION: 1.0
# 
# AUTHOR: Miguel Ibarra (miguelib@ufl.edu)
#
# DESCRIPTION: This program creates a distribution plot by features for a given datset
#
#######################################################################################

# Built-in packages
import os
import copy
import logging
import argparse
from argparse import RawDescriptionHelpFormatter

# Add-on packages
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Local Packages
import logger as sl
import module_box as box
import module_distribution as dis
from interface import wideToDesign
from manager_color import colorHandler
from manager_figure import figureHandler

def getOptions():
    """
    Function to pull in arguments
    """
    description = """ Distribution Analysis: Plot sample distrubtions. """
    parser = argparse.ArgumentParser(description=description, formatter_class=
                                    RawDescriptionHelpFormatter)
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

    output = parser.add_argument_group(title="Output paths", 
                                        description="Paths and outputs")
    output.add_argument("-f","--figure", dest="figure", action='store',
                        required=True, help="Output figure name [pdf].")

    args = parser.parse_args()
    return(args)

def main(args):
    """
    Function to call all other functions
    """
    # Loading files with interface
    dat = wideToDesign(args.input,args.design,args.uniqID,group=args.group)
    logger.info(u"""Data loaded by interface""")

    # Subseting wide to get features for wide files with more that 50 features
    if len(dat.wide.index) > 50:
        wide = dat.wide.sample(n=50,axis=0)
        wide = wide.T
    else:
        wide = dat.wide.T

    # Set palette of colors
    palette = colorHandler(pal="tableau", col="Tableau_20")
    logger.info(u"""Using tableau Tableau_20 palette""")

    # Get colors for each feature
    colors = palette.getColorByIndex(dataFrame=wide.T)

    # Stablishing figure layout (x,y,colspan,rowspan)
    axisLayout = [(0,0,1,3),(3,0,1,1)]

    # Iterating over groups
    if args.group:

        # Saving figure
        with PdfPages(os.path.abspath(args.figure)) as pdf:

            #iterating over groups
            for name, group in dat.design.groupby(args.group):

                # Creating a figure template
                figure = figureHandler(proj='2d', numAx=2, numRow=4, numCol=1, 
                                        figsize=(8,13), arrangement=axisLayout)
                # Adding figure Title
                figure.formatAxis(figTitle="Distribution by Features {0}".format(name),
                                xlim="ignore", ylim="ignore",axnum=0,showX=False)

                # Plotting boxPlot
                logger.info(u"Plotting boxplot {0}".format(name))
                box.boxDF(ax=figure.ax[0], colors=colors, dat=wide.T[group.index].T,
                         vert=False,rot=0)

                # Plotting density plot
                logger.info(u"Plotting density plot {0}".format(name))
                dis.plotDensityDF(data=wide.T[group.index].T.unstack(), 
                                ax=figure.ax[1], colors=colors[0])

                # Adding figure to pdf object
                figure.addToPdf(pdf)

            # Creating axisLayout for figure and figure object
            # (x,y,colspan,rowspan)
            figure = figureHandler(proj='2d', numAx=2, numRow=4, numCol=1, 
                                    figsize=(8,13), arrangement=axisLayout)

            # Adding figure Title
            figure.formatAxis(figTitle="Distribution by Features All",xlim="ignore",
                                ylim="ignore",axnum=0,showX=False)

            # Printing boxPlot
            logger.info(u"Plotting Boxplot All")
            box.boxDF(ax=figure.ax[0], colors=colors, dat=wide, vert=False,rot=0)

            # Plot density plot
            logger.info(u"Plotting Density Plot All")
            dis.plotDensityDF(data=dat.wide.T.unstack(), ax=figure.ax[1], colors=colors[0])

            # Adding figure to pdf object
            figure.addToPdf(pdf)

            #Ending script
            logger.info(u"Ending script")

if __name__ == '__main__':
    # Command line options
    args = getOptions()

    # Setting logging
    logger = logging.getLogger()
    sl.setLogger(logger)

    # Print logger info
    logger.info("Importing data with following parameters: "\
            "\n\tWide: {0}"\
            "\n\tDesign: {1}"\
            "\n\tUnique ID: {2}".format(args.input, args.design, args.uniqID))


    # Main
    main(args)