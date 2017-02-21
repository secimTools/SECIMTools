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

#Future imports
from __future__ import division

# Built-in packages
import os
import math
import logging
import argparse
from itertools import combinations
from argparse import RawDescriptionHelpFormatter

# Add-on packages
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages

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
                        required=True, help="Name of the column with unique"\
                        " identifiers.")
    standard.add_argument("-g", "--group",dest="group", action='store', 
                        required=False,  default=False, help="Name of the"\
                        " column with groups.")

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
                        required=False, help="Name of output file to store"\
                        "scatter plots for 3 principal components.")

    plot = parser.add_argument_group(title='Plot options')
    plot.add_argument("-pal","--palette",dest="palette",action='store',required=False, 
                        default="tableau", help="Name of the palette to use.")
    plot.add_argument("-col","--color",dest="color",action="store",required=False, 
                        default="Tableau_20", help="Name of a valid color scheme"\
                        " on the selected palette")

    args = parser.parse_args()
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

    # Create summay frame
    summary = np.array([sd,var,cumVar]).T

    return scores,loadings,summary

def plotScatterplot2D(data,pdf,dat,nloads=3):
    """
    Plots Scatterplots 2D for a number of loadngs for PCA.

    :Arguments:
        :type data: pandas.DataFrame
        :param data: Loadings of the PCA.

        :type pdf: pdf object
        :param pdf: PDF object to save all the generated figures.

        :type dat: interface.wideToDesign
        :param dat: wideToDesign instance providing acces to all input data.

        :type nloads: int
        :param nloads: Number of principal components to create pairwise combs.
    """
    # Get colors
    if  dat.group:
        colorDes,ucGroups,combName=palette.getColors(design=dat.design,
                                                    groups=[dat.group])
    else:
        colorList = list(palette.mpl_colors[0]) #First color on the palette
        ucGroups = dict()

    # Selecting amount of pairwise combinations to plot scaterplots for loads.
    for x, y in list(combinations(data.columns.tolist()[:nloads],2)):

        # Create a single-figure figure handler object
        fh = figureHandler(proj="2d", figsize=(14,8))

        # Create a title for the figure
        title = "{0} vs {1}".format(x,y)

        # Get specific color list
        if dat.group:
            colorList = [ucGroups[dat.design[dat.group][idx]] for idx in data[x].index]

        # Plot the scatterplot based on data
        scatter.scatter2D(x=list(data[x]), y=list(data[y]),
                         colorList=colorList, ax=fh.ax[0])

        # Create legend
        fh.makeLegend(ax=fh.ax[0],ucGroups=ucGroups,group=dat.group)

        # Shrink axis to fit legend
        fh.shrink()

        # Despine axis
        fh.despine(fh.ax[0])

        # Formatting axis
        fh.formatAxis(figTitle=title,xTitle="Scores on {0}".format(x),
            yTitle="Scores on {0}".format(y),grid=False)

        # Adding figure to pdf
        fh.addToPdf(dpi=600,pdfPages=pdf)

def plotScatterplot3D(data,pdf,dat):
    """
    Plots Scatterplots 3D for a given number of loadngs for PCA.

    :Arguments:
        :type data: pandas.DataFrame
        :param data: Loadings of the PCA.

        :type pdf: pdf object
        :param pdf: PDF object to save all the generated figures.

        :type dat: interface.wideToDesign
        :param dat: wideToDesign instance providing acces to all input data.
    """
    
    # Open figure handler with 3D projection
    fh = figureHandler(proj="3d", figsize=(14,8))

    # Getting colors by group
    if dat.group:
        colorDes,ucGroups,combName=palette.getColors(design=dat.design,
                                                    groups=[dat.group])
        colorList = colorDes.colors.tolist()
    else:
        colorList = list(palette.mpl_colors[0]) #First color on the palette
        ucGroups = dict()
    
    # Plot scatterplot3D 
    ax = scatter.scatter3D(ax=fh.ax[0], colorList=colorList,
                        x=list(data["PC1"]), y=list(data["PC2"]),
                        z=list(data["PC3"]))

    # Make legends
    if dat.group:
        fh.makeLegend(ax=fh.ax[0],ucGroups=ucGroups,group=dat.group)

    # Add Titles to the PCA
    fh.format3D(xTitle="PC1",yTitle="PC2",zTitle="PC3")

    # Add Figure to the PDf
    fh.addToPdf(dpi=600, pdfPages=pdf)

def main(args):
    #Loading data trought Interface
    dat = wideToDesign(args.input, args.design, args.uniqID, group=args.group, 
                        logger=logger)

    # Cleaning from missing data
    dat.dropMissing()

    # Transpossing matrix
    dat.wide = dat.wide.T

    # RunPCA
    scores, loadings, summary = runPCA(dat.wide)

    # Create name of the components
    header = ["PC{0}".format(x+1) for x in range(summary.shape[0])]

    # Convert loadings and scores to Pandas DataFrame rename index in summary
    scores   = pd.DataFrame(data=scores, index=dat.wide.index, columns=header)
    loadings = pd.DataFrame(data=loadings, index=header, columns=dat.wide.columns)
    summary  = pd.DataFrame(data=summary, index=header, columns=["standar_deviation",
                "proportion_of_variance_explained","cumulative_proportion_of_variance_explained"])

    # Save Scores, Loadings and Summary
    scores.to_csv(args.score_out, sep="\t", index_label='sampleID')
    loadings.to_csv(args.load_out, sep="\t", index_label=dat.uniqID)
    summary.to_csv(args.summary_out, sep="\t", index_label="PCs")

    #Plotting scatter plot 3D
    logger.info(u"Plotting PCA scores")
    with PdfPages(os.path.abspath(args.figure)) as pdfOut:
        plotScatterplot2D(data=scores, pdf=pdfOut, dat=dat)
        plotScatterplot3D(data=scores, pdf=pdfOut, dat=dat)

    #Ending script
    logger.info(u"Finishing running of PCA")


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

    # Run Main Script
    main(args)
