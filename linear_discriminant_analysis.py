#!/usr/bin/env python
################################################################################
# DATE: 2017/02/22
#
# SCRIPT: linear_discriminant_analysis.py
#
# VERSION: 1.0
# 
# AUTHOR: Miguel A. Ibarra (miguelib@ufl.edu)
# 
# DESCRIPTION: This script takes a a wide format file and performs a linear
# discriminant analysis.
#
################################################################################

#Future imports
from __future__ import division

# Built-in packages
import os
import math
import logging
import argparse
import warnings
from itertools import combinations
from argparse import RawDescriptionHelpFormatter

# Add-on packages
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Local Packages
import logger as sl
import module_scatter as scatter
from interface import wideToDesign
from manager_color import colorHandler
from manager_figure import figureHandler

def getOptions(myopts=None):
    """ Function to pull in arguments """
    description="""  
    This script runs a Linear Discriminant Analysis (LDA)
    """
    
    parser = argparse.ArgumentParser(description=description, 
                                    formatter_class=RawDescriptionHelpFormatter)
    standard = parser.add_argument_group(title='Standard input', 
                                description='Standard input for SECIM tools.')
    standard.add_argument( "-i","--input", dest="input", action='store', 
                        required=True, help="Input dataset in wide format.")
    standard.add_argument("-d" ,"--design",dest="design", action='store', 
                        required=True, help="Design file.")
    standard.add_argument("-id", "--ID",dest="uniqID", action='store', required=True, 
                        help="Name of the column with unique identifiers.")
    standard.add_argument("-g", "--group",dest="group", action='store', required=True, 
                        default=False,help="Name of the column with groups.")
    standard.add_argument("-l","--levels",dest="levels",action="store", 
                        required=False, default=False, help="Different groups to"\
                        " sort by separeted by commas.")

    optional = parser.add_argument_group(title='Optional input', 
                            description='Optional/Specific input for the tool.')
    optional.add_argument("-nc", "--nComponents",dest="nComponents", action='store',
                        type= int, required=False, default=None,  
                        help="Number of component s[Default == 2].")

    output = parser.add_argument_group(title='Required output')
    output.add_argument("-o","--out",dest="out",action='store',required=True, 
                        help="Name of output file to store scores. TSV format.")
    output.add_argument("-f","--figure",dest="figure",action="store",required=True,
                        help="Name of output file to store scatter plots for scores")

    plot = parser.add_argument_group(title='Plot options')
    plot.add_argument("-pal","--palette",dest="palette",action='store',required=False, 
                        default="tableau", help="Name of the palette to use.")
    plot.add_argument("-col","--color",dest="color",action="store",required=False, 
                        default="Tableau_20", help="Name of a valid color scheme"\
                        " on the selected palette")
    args = parser.parse_args()

    # Standardized output paths
    args.figure = os.path.abspath(args.figure)
    args.out    = os.path.abspath(args.out)

    # Split levels if levels
    if args.levels:
        args.levels = args.levels.split(",")

    return(args)

def runLDA(dat,nComp):
    """
    Runs LDA over a wide formated dataset

    :Arguments:
        :type dat: interface.wideToDesign
        :param dat: wideToDesign instance providing acces to all input data.

        :type nComp: int
        :param nComp: Number of components to use, if None the program will
                        calculate n_groups -1 .
    :Returns:
        :rtype scores_df: pandas.DataFrame
        :return scores_df: Scores of the LDA.
    """
    #Initialize LDA class and stablished number of coponents
    lda = LinearDiscriminantAnalysis(n_components=nComp)

    #Get scores of LDA (Fit LDA)
    scores = lda.fit_transform(dat.wide.T,dat.transpose()[dat.group])

    # If no number the componentes specified asign the max number of componentes
    # calculated by the meythod.
    if nComp is None:
        nComp = scores.shape[1]

    # Create column names for scores file
    LVNames = ["Component_{0}".format(i+1) for i in range(nComp)]

    # Create padas dataframe for scores
    scores_df = pd.DataFrame(scores,columns=LVNames,index=dat.wide.columns.values)

    # Return scores for LDA
    return scores_df

def plotScores(data, palette, pdf):
    """
    Runs LDA over a wide formated dataset

    :Arguments:
        :type data: pandas.DataFrame
        :param data: Scores of the LDA.

        :type palette: colorManager.object
        :param palette: Object from color manager

        :type pdf: pdf object
        :param pdf: PDF object to save all the generated figures.

    :Returns:
        :rtype scores_df: pandas.DataFrame
        :return scores_df: Scores of the LDA.
    """
    # Create a scatter plot for each combination of the scores 
    for x, y in list(combinations(data.columns.tolist(),2)):

        # Create a single-figure figure handler object
        fh = figureHandler(proj="2d", figsize=(14,8))

        # Create a title for the figure
        title = "{0} vs {1}".format(x,y)

        # Plot the scatterplot based on data
        scatter.scatter2D(x=list(data[x]), y=list(data[y]),
                         colorList=palette.design.colors.tolist(), ax=fh.ax[0])

        # Create legend
        fh.makeLegend(ax=fh.ax[0],ucGroups=palette.ugColors,group=palette.combName)

        # Shrink axis to fit legend
        fh.shrink()

        # Despine axis
        fh.despine(fh.ax[0])

        # Formatting axis
        fh.formatAxis(figTitle=title,xTitle="Scores on {0}".format(x),
            yTitle="Scores on {0}".format(y),grid=False)

        # Adding figure to pdf
        fh.addToPdf(dpi=600,pdfPages=pdf)

def main(args):
    # Checking if levels
    if args.levels and args.group:
        levels = [args.group]+args.levels
    elif args.group and not args.levels:
        levels = [args.group]
    else:
        levels = []

    #Loading data trought Interface
    dat = wideToDesign(args.input, args.design, args.uniqID,group=args.group, 
            anno=args.levels, logger=logger)

    # Treat everything as numeric
    dat.wide = dat.wide.applymap(float)
    
    # Cleaning from missing data
    dat.dropMissing()

    # Get colors for each sample based on the group
    palette.getColors(design=dat.design, groups=levels)

    #Run LDA
    logger.info(u"Runing LDA on data")
    scores_df = runLDA(dat, nComp=args.nComponents)
 
    # Plotting scatter plot for scores
    logger.info(u"Plotting LDA scores")
    with PdfPages(args.figure) as pdfOut:
        plotScores(data=scores_df, palette=palette, pdf=pdfOut)

    # Save scores
    scores_df.to_csv(args.out,sep="\t",index_label="sampleID")

    #Ending script
    logger.info(u"Finishing running of LDA")

if __name__ == '__main__':   
    # Command line options
    args = getOptions()

    #Set logger
    logger = logging.getLogger()
    sl.setLogger(logger)

    #Starting script
    logger.info(u"""Importing data with following parameters:
                Input: {0}
                Design: {1}
                uniqID: {2}
                group: {3}
                """.format(args.input, args.design, args.uniqID, args.group))

    # Stablishing color palette
    palette = colorHandler(pal=args.palette, col=args.color)
    logger.info(u"Using {0} color scheme from {1} palette".format(args.color,
                args.palette))

    # Getting rid of warnings.
    warnings.filterwarnings("ignore")

    # Main code
    main(args)
