#!/usr/bin/env python
################################################################################
# DATE: 2017/02/21
#
# SCRIPT: principal_component_analysis.py
#
# VERSION: 2.2
# 
# AUTHOR: Coded by: Miguel A Ibarra (miguelib@ufl.edu) 
#         Edited by: Matt Thoburn (mthoburn@ufl.edu)
#         Last review  by: Miguel A Ibarra (miguelib@ufl.edu) 
# 
# DESCRIPTION: This script takes a a wide format file and makes a principal 
#   component analysis (PCA) on it.
#
################################################################################
# Import future libraries
from __future__ import division

# Import built-in libraries
import os
import math
import logging
import argparse
from itertools import combinations
from argparse import RawDescriptionHelpFormatter

# Import add-on libraries
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages

# Import local data libraries
from secimtools.dataManager import logger as sl
from secimtools.dataManager.interface import wideToDesign

# Import local plotting libraries
from secimtools.visualManager import module_scatter as scatter
from secimtools.visualManager.manager_color import colorHandler
from secimtools.visualManager.manager_figure import figureHandler

def getOptions(myopts=None):
    """ Function to pull in arguments """
    description="""  
    This script runs a Principal Component Analysis (PCA)

    """
    parser = argparse.ArgumentParser(description=description, 
                                    formatter_class=RawDescriptionHelpFormatter)
    # Standard Input
    standard = parser.add_argument_group(title='Standard input', 
                                description='Standard input for SECIM tools.')
    standard.add_argument( "-i","--input", dest="input", action='store', 
                        required=True, help="Input dataset in wide format.")
    standard.add_argument("-d" ,"--design",dest="design", action='store', 
                        required=True, help="Design file.")
    standard.add_argument("-id", "--ID",dest="uniqID", action='store', 
                        required=True, help="Name of the column with unique"\
                        " identifiers.")
    standard.add_argument("-g", "--group",dest="group", action='store', 
                        required=False,  default=False, help="Name of the"\
                        " column with groups.")
    standard.add_argument("-l","--levels",dest="levels",action="store", 
                        required=False, default=False, help="Different groups to"\
                        " sort by separeted by commas.")
    # Tool output
    output = parser.add_argument_group(title='Required output')
    output.add_argument("-lo","--load_out",dest="load_out",action='store',
                        required=True, help="Name of output file to store "\
                        "loadings. TSV format.")
    output.add_argument("-so","--score_out",dest="score_out",action='store',
                        required=True, help="Name of output file to store "\
                        "scores. TSV format.")
    output.add_argument("-su","--summary_out",dest="summary_out",action='store',
                        required=True, help="Name of output file to store "\
                        "summaries of the components. TSV format.")
    output.add_argument("-f","--figure",dest="figure",action="store",
                        required=True, help="Name of output file to store"\
                        "scatter plots for 3 principal components.")
    # Plot options
    plot = parser.add_argument_group(title='Plot options')
    plot.add_argument("-pal","--palette",dest="palette",action='store',required=False, 
                        default="tableau", help="Name of the palette to use.")
    plot.add_argument("-col","--color",dest="color",action="store",required=False, 
                        default="Tableau_20", help="Name of a valid color scheme"\
                        " on the selected palette")
    args = parser.parse_args()

    # Standardized output paths
    args.figure      = os.path.abspath(args.figure) 
    args.load_out    = os.path.abspath(args.load_out)
    args.score_out   = os.path.abspath(args.score_out)
    args.summary_out = os.path.abspath(args.summary_out)

    # Split levels if levels
    if args.levels:
        args.levels = args.levels.split(",")

    return(args)

def runPCA(wide):
    """
    Runs PCA over a wide formated dataset

    :Arguments:
        :type wide: pandas.core.frame.DataFrame
        :param wide: DataFrame with the wide file data

    :Returns:
        :rtype loadings: numpy.ndarray
        :return loadings: Loads of the PCA

        :rtype scores: numpy.ndarray
        :return scores: Scores of the PCA

        :rtype block: pandas.core.frame.DataFrame
        :return block: Useful information derived by the PCA STD, Proportion of 
                        the variance explained by this variable and acumulative
                        proportion of the variance explained.
    """
    logger.info(u"Runing PCA on data")
    
    # Scalling data
    wide_s = wide.copy()
    wide_s["var"] = wide.std(axis=1)
    wide_s = wide_s.apply(lambda x: x/math.sqrt(x["var"]),axis=1)
    wide_s = wide_s.drop("var", axis=1)

    #Initialize PCA class with default values
    pca = PCA()

    #Get scores of PCA (Fit PCA)
    scores = pca.fit_transform(wide)

    #Get loadings
    loadings = pca.components_

    #Get aditional information out of PCA (summary)
    sd     = scores.std(axis=0)
    var    = pca.explained_variance_ratio_
    cumVar = var.cumsum()

    # Create summay file
    summary = np.array([sd,var,cumVar]).T

    # Create headers 
    header = ["PC{0}".format(x+1) for x in range(summary.shape[0])]

    # Convert loadings, scores and summaries to Pandas DataFrame rename index
    df_scores   = pd.DataFrame(data=scores, index=wide.index, columns=header)
    df_loadings = pd.DataFrame(data=loadings, index=header, columns=wide.columns)
    df_summary  = pd.DataFrame(data=summary, index=header, columns=["standard_deviation",
                "proportion_of_variance_explained","cumulative_proportion_of_variance_explained"])

    return df_scores, df_loadings, df_summary

def plotScatterplot2D(data, palette, pdf, nloads=3):
    """
    Plots Scatterplots 2D for a number of loadngs for PCA.

    :Arguments:
        :type data: pandas.DataFrame
        :param data: Loadings of the PCA.

        :type pdf: pdf object
        :param pdf: PDF object to save all the generated figures.

        :type nloads: int
        :param nloads: Number of principal components to create pairwise combs.
    """

    # Selecting amount of pairwise combinations to plot scaterplots for loads.
    for x, y in list(combinations(data.columns.tolist()[:nloads],2)):

        # Create a single-figure figure handler object
        fh = figureHandler(proj="2d", figsize=(14,8))

        # Create a title for the figure
        title = "{0} vs {1}".format(x,y)

        # Plot the scatterplot based on data
        scatter.scatter2D(x=list(data[x]), y=list(data[y]),
                         colorList=palette.design.colors.tolist(), ax=fh.ax[0])

        # Create legend
        fh.makeLegend(ax=fh.ax[0], ucGroups=palette.ugColors, group=palette.combName)

        # Shrink axis to fit legend
        fh.shrink()

        # Despine axis
        fh.despine(fh.ax[0])

        # Formatting axis
        fh.formatAxis(figTitle=title,xTitle="Scores on {0}".format(x),
            yTitle="Scores on {0}".format(y),grid=False)

        # Adding figure to pdf
        fh.addToPdf(dpi=600,pdfPages=pdf)

def plotScatterplot3D(data, palette, pdf):
    """
    Plots Scatterplots 3D for a given number of loadngs for PCA.

    :Arguments:
        :type data: pandas.DataFrame
        :param data: Loadings of the PCA.

        :type pdf: pdf object
        :param pdf: PDF object to save all the generated figures.
    """
    
    # Open figure handler with 3D projection
    fh = figureHandler(proj="3d", figsize=(14,8))

    # Plot scatterplot3D 
    ax = scatter.scatter3D(ax=fh.ax[0], colorList=palette.design.colors.tolist(),
                        x=list(data["PC1"]), y=list(data["PC2"]),
                        z=list(data["PC3"]))

    # Make legends
    fh.makeLegend(ax=fh.ax[0], ucGroups=palette.ugColors, group=palette.combName)

    # Add Titles to the PCA
    fh.format3D(xTitle="PC1",yTitle="PC2",zTitle="PC3")

    # Add Figure to the PDf
    fh.addToPdf(dpi=600, pdfPages=pdf)

def main(args):
    # Checking if levels
    if args.levels and args.group:
        levels = [args.group]+args.levels
    elif args.group and not args.levels:
        levels = [args.group]
    else:
        levels = []

    #Loading data trought Interface
    dat = wideToDesign(args.input, args.design, args.uniqID, group=args.group, 
                        anno=args.levels, logger=logger)

    # Cleaning from missing data
    dat.dropMissing()

    # Get colors for each sample based on the group
    palette.getColors(design=dat.design, groups=levels)

    # Transpossing matrix
    dat.wide = dat.wide.T

    # RunPCA
    df_scores, df_loadings, df_summary = runPCA(dat.wide)

    #Plotting scatter plot 3D
    logger.info(u"Plotting PCA scores")
    with PdfPages(args.figure) as pdfOut:
        plotScatterplot2D(data=df_scores, palette=palette, pdf=pdfOut)
        plotScatterplot3D(data=df_scores, palette=palette, pdf=pdfOut)

    # Save Scores, Loadings and Summary
    df_scores.to_csv(args.score_out, sep="\t", index_label='sampleID')
    df_loadings.to_csv(args.load_out, sep="\t", index_label=dat.uniqID)
    df_summary.to_csv(args.summary_out, sep="\t", index_label="PCs")

    #Ending script
    logger.info(u"Finishing running of PCA")

if __name__ == '__main__':
    # Command line options
    args = getOptions()

    # Set logger
    logger = logging.getLogger()
    sl.setLogger(logger)

    # Starting script
    logger.info("Importing data with following parameters:"\
                "\n\tInput: {0}"\
                "\n\tDesign: {1}"\
                "\n\tuniqID: {2}"\
                "\n\tgroup: {3}".format(args.input, args.design, args.uniqID, 
                                args.group))

    # Stablishing color palette
    palette = colorHandler(pal=args.palette, col=args.color)
    logger.info(u"Using {0} color scheme from {1} palette".format(args.color,
                args.palette))

    # Run Main Script
    main(args)
