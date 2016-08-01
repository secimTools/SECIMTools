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
import logging
import argparse
from argparse import RawDescriptionHelpFormatter

# Add-on packages
import pandas as pd

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
    dat = wideToDesign(args.input,args.design,args.uniqID)
    logger.info(u"""Data loaded by interface""")
    
    # Subseting wide to get features for wide files with more that 50 features
    if len(dat.wide.index) > 50:
        wide = dat.wide.sample(n=50,axis=0)
        wide = wide.T
    else:
        wide = dat.wide.T

    # Creating axisLayout for figure and figure object
    # (x,y,colspan,rowspan)
    axisLayout = [(0,0,1,3),(3,0,1,1)]
    figure = figureHandler(proj='2d', numAx=2, numRow=4, numCol=1, figsize=(8,15),
                            arrangement=axisLayout)
    logger.info(u"""Figure created""")

    # Adding figure Title
    figure.formatAxis(figTitle="Distribution by Features",xlim="ignore",
                        ylim="ignore",axnum=0,showX=False)

    # Set palette of colors
    palette = colorHandler(pal="tableau", col="Tableau_20")
    logger.info(u"""Using tableaau Tableau_20 palette""")

    # Get colors for each feature
    colors = palette.getColorByIndex(dataFrame=wide.T)

    # Printing boxPlot
    box.boxDF(ax=figure.ax[0], colors=colors, dat=wide, vert=False,rot=0)
    logger.info(u"""Plotting boxplot""")

    # Converting wide dataframe to series
    series=wide.unstack()

    # Plot density plot
    dis.plotDensityDF(data=wide.unstack(), ax=figure.ax[1], colors=colors[0])
    logger.info(u"""Plotting density plot""")

    # Shirnk figure
    figure.shrink()

    # Saving figure
    figure.export(dpi=600,out=args.figure)
    logger.info(u"""Ending script""")

if __name__ == '__main__':
    # Command line options
    args = getOptions()

    # Setting logging
    logger = logging.getLogger()
    sl.setLogger(logger)

    # Print logger info
    logger.info(u"""Importing data with following parameters: 
            \n\tWide: {0}
            \n\tDesign: {1}
            \n\tUnique ID: {2}
            """ .format(args.input, args.design, args.uniqID))


    # Main
    main(args)