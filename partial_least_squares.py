#!/usr/bin/env python
################################################################################
# DATE: 2016/May/06, rev: 2016/June/03
#
# SCRIPT: pca.py
#
# VERSION: 1.1
# 
# AUTHOR: Miguel A Ibarra (miguelib@ufl.edu) Edited by: Matt Thoburn (mthoburn@ufl.edu)
# 
# DESCRIPTION: This script takes a a wide format file and makes a principal 
#   component analysis (PCA) on it.
#
################################################################################

#Future imports
from __future__ import division

# Built-in packages
import os
import math
import logging
import argparse
import itertools
from argparse import RawDescriptionHelpFormatter

# Add-on packages
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cross_decomposition import PLSRegression

# Local Packages
import logger as sl
import module_scatter as scatter
from interface import wideToDesign
from manager_color import colorHandler
from manager_figure import figureHandler

def getOptions(myopts=None):
    """ Function to pull in arguments """
    description="""  
    This script runs a Principal Component Analysis (PCA)

    """
    parser = argparse.ArgumentParser(description=description, 
                                    formatter_class=RawDescriptionHelpFormatter)
    standard = parser.add_argument_group(title='Standard input', 
                                description='Standard input for SECIM tools.')
    standard.add_argument( "-i","--input", dest="input", action='store', 
                        required=True, help="Input dataset in wide format.")
    standard.add_argument("-d" ,"--design",dest="design", action='store', 
                        required=True, help="Design file.")
    standard.add_argument("-id", "--ID",dest="uniqID", action='store',
                        required=True,  help="Name of the column with unique"\
                        " identifiers.")
    standard.add_argument("-g", "--group",dest="group", action='store',
                        required=True, default=False, help="Name of the column"\
                        " with groups.")
    standard.add_argument("-l","--levels",dest="levels",action="store", 
                        required=False, default=False, help="Different groups to"\
                        " sort by separeted by commas.")

    tool = parser.add_argument_group(title='Tool specific input', 
                                description='Input specific for this tool.')
    tool.add_argument("-t", "--toCompare",dest="toCompare", action='store', 
                        required=True, default=True, help="Name of"
                        " the elements to compare in group col.")
    tool.add_argument("-n", "--nComp",dest="nComp", action='store', 
                        required=False,  default=3, type = int, help="Number"\
                        " of components.")

    output = parser.add_argument_group(title='Required output')
    output.add_argument("-os","--outScores",dest="outScores",action='store',required=True, 
                        help="Name of output file to store loadings. TSV format.")
    output.add_argument("-ow","--outWeights",dest="outWeights",action='store',required=True,
                        help="Name of output file to store weights. TSV format.")
    output.add_argument("-f","--figure",dest="figure",action="store",required=False,
                        help="Name of output file to store scatter plots for scores")

    plot = parser.add_argument_group(title='Plot options')
    plot.add_argument("-pal","--palette",dest="palette",action='store',required=False, 
                        default="tableau", help="Name of the palette to use.")
    plot.add_argument("-col","--color",dest="color",action="store",required=False, 
                        default="Tableau_20", help="Name of a valid color scheme"\
                        " on the selected palette")

    development = parser.add_argument_group(title='Development Settings')
    development.add_argument("--debug",dest="debug",action='store_true',
                        required=False,help="Add debugging log output.")
    args = parser.parse_args()

    # Standardize Paths
    args.figure     = os.path.abspath(args.figure)
    args.outScores  = os.path.abspath(args.outScores)
    args.outWeights = os.path.abspath(args.outWeights)

    # Split levels if levels
    if args.levels:
        args.levels = args.levels.split(",")

    # Split to compare
    args.toCompare = args.toCompare.split(",")

    return(args)

def runPLS(trans, group,toCompare,nComp):
    """
    Runs PCA over a wide formated dataset

    :Arguments:
        :type dat: pandas.core.frame.DataFrame
        :param dat: DataFrame with the wide file data

    :Returns:
        :rtype scores: pandas.DataFrame
        :return scores: Scores of the PCA

        :rtype weights: pandas.DataFrame
        :return weights: weights of the PCA
    """
    logger.info(u"Runing PLS on data")
    # Subsetting data to just have the value we're interested in
    subset = [subs for name,subs in trans.groupby(group) if name in toCompare]
    subset = pd.concat(subset,axis=0)

    # Creating pseudolinear Y value
    Y = np.where(subset[group]==toCompare[0],int(1),int(0));subset

    # Remove group column from the subseted data
    subset.drop(group, axis=1, inplace=True)

    # Apply PLS-DA model to our data, we neet to specify the number of latent
    # variables (LV), to use for our data.
    plsr = PLSRegression(n_components=nComp, scale=False)

    # Fit the model
    plsr.fit(subset.values,Y)

    # Creating column names for scores and weights
    LVNames = ["LV_{0}".format(i+1) for i in range(nComp)]

    # The scores describe the position of each sample in each determined LV.
    # Columns for LVs and rows for Samples. (These are the ones the you plot).
    # Generrates a dataFrame with the scores.
    scores_df  = pd.DataFrame(plsr.x_scores_,index=subset.index, columns=LVNames)

    # The weights describe the contribution of each variable to each LV. The
    # columns are for the LVs and the rows are for each feature.
    # generates a dataFrame with the weights.
    weights_df = pd.DataFrame(plsr.x_weights_,index=subset.columns, columns=LVNames)

    # Returning the results
    return scores_df,weights_df

def plotScores(data, palette, pdf):
    """
    This function creates a PDF file with 3 scatter plots for the combinations 
    of the 3 principal components. PC1 vs PC2, PC1 vs PC3, PC2 vs PC3.

    :Arguments:
        :type data: pandas.core.frame.DataFrame
        :param data: Data frame with the data to plot.
        
        :type outpath: string
        :param outpath: Path for the output file

        :type group: string
        :param group: Name of the column that contains the group information on the design file.

    :Return:
        :rtype PDF: file
        :retrn PDF: file with the 3 scatter plots for PC1 vs PC2, PC1 vs PC3, PC2  vs PC3.
    """
    for x,y in  list(itertools.combinations(data.columns.tolist(),2)):
        # Creating a figure handler object
        fh = figureHandler(proj="2d", figsize=(14,8))

        # Creating title for the figure
        title = "{0} vs {1}".format(x,y)
        
        # Creating the scatterplot 2D
        scatter.scatter2D(ax=fh.ax[0], x=list(data[x]), y=list(data[y]),
                        colorList=palette.design.colors.tolist())

        # Despine axis
        fh.despine(fh.ax[0])

        # Print Legend
        fh.makeLegend(ax=fh.ax[0], ucGroups=palette.ugColors, group=palette.combName)

        # Shinking the plot so everything fits
        fh.shrink()

        # Format Axis
        fh.formatAxis(figTitle=title, xTitle="Scores on {0}".format(x),
                    yTitle="Scores on {0}".format(y), grid=False)

        # Adding figure to pdf
        fh.addToPdf(dpi=90,pdfPages=pdf)

def main(args):
    # Checking if levels
    if args.levels and args.group:
        levels = [args.group]+args.levels
    elif args.group and not args.levels:
        levels = [args.group]
    else:
        levels = []

    # Loading data trought Interface
    dat = wideToDesign(args.input, args.design, args.uniqID, group=args.group, 
                        anno=args.levels, logger=logger)

    # Treat everything as numeric
    dat.wide = dat.wide.applymap(float)

    # Cleaning from missing data
    dat.dropMissing()

    # Get colors for each sample based on the group
    palette.getColors(design=dat.design, groups=levels)

    # Transpose data
    dat.trans = dat.transpose()

    # Run PLS
    df_scores,df_weights = runPLS(dat.trans,dat.group,args.toCompare,args.nComp)

    # Update palette afterdrop selection of groups toCompare
    palette.design   =  palette.design.T[df_scores.index].T
    palette.ugColors =  {ugc:palette.ugColors[ugc] for ugc in palette.ugColors.keys() if ugc in args.toCompare}

    # Plotting scatter plot for scores
    with PdfPages(args.figure) as pdfOut:
        logger.info(u"Plotting PLS scores")
        plotScores(data=df_scores, palette=palette, pdf=pdfOut)

    # Save df_scores and df_weights  to tsv files
    df_scores.to_csv(args.outScores,  sep="\t", index_label='sampleID')
    df_weights.to_csv(args.outWeights,sep="\t", index_label=dat.uniqID)

    #Ending script
    logger.info(u"Finishing running of PLS")

if __name__ == '__main__':
    # Command line options
    args = getOptions()

    #Set logger
    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel="debug")
    else:
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

    # running main script
    main(args)
