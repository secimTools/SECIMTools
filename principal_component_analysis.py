#!/usr/bin/env python
################################################################################
# DATE: 2016/10/31
#
# SCRIPT: pca.py
#
# VERSION: 1.1
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
    output.add_argument("-lo","--load_out",dest="lname",action='store',
                        required=True, help="Name of output file to store "\
                        "loadings. TSV format.")
    output.add_argument("-so","--score_out",dest="sname",action='store',
                        required=True, help="Name of output file to store "\
                        "scores. TSV format.")
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

def dropMissing(wide):
    """
    Drops missing data out of the wide file

    :Arguments:
        :type wide: pandas.core.frame.DataFrame
        :param wide: DataFrame with the wide file data

    :Returns:
        :rtype wide: pandas.core.frame.DataFrame
        :return wide: DataFrame with the wide file data without missing data
    """
    #Warning
    logger.warn("Missing values were found")

    #Count of original
    nRows = len(wide.index)      

    #Dropping 
    wide.dropna(inplace=True)    

    #Count of dropped
    nRowsNoMiss = len(wide.index)  

    #Warning
    logger.warn("{} rows were dropped because of missing values.".
                format(nRows - nRowsNoMiss))
    return wide

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
    #Initialize PCA class with default values
    pca = PCA()

    #Get scores of PCA (Fit PCA)
    scores = pca.fit_transform(wide)

    #Get loadings
    if  len(wide.columns)>len(wide.index):
        loadings = pca.components_.T
    else:
        loadings = pca.components_

    #Get aditional information out of PCA, this will mimic the output from R
    sd = scores.std(axis=0)
    propVar = pca.explained_variance_ratio_
    cumPropVar = propVar.cumsum()

    #Crete labels(index) for the infor
    labels = np.array(['#Std. deviation', '#Proportion of variance explained',
                        '#Cumulative propowidertion of variance explained'])

    #Create dataFrame for block output
    block = pd.DataFrame([sd, propVar, cumPropVar],index=labels)

    return loadings,scores,block

def createOutput(item,index):
    """
    Runs PCA over a wide formated dataset

    :Arguments:
        :type item: numpy.ndarray
        :param item: Contains the data eiter loads or scores

        :type index: numpy.ndarray
        :param index: Contains the index for each item, could either be 
                        column names (sampleID) or row names (rowID).
    :Returns:
        :rtype itemOut: pandas.core.frame.DataFrame
        :return itemOut: data frame with full output structure containing item,
                         index and block.
    """

    logger.info(u"Creating output file")
    #Create header
    header = np.array(['PC{}'.format(x + 1) for x in range(item.shape[1])])

    #Creating dataframe
    itemOut=pd.DataFrame(data=item,index=index,columns=header)

    #Return 
    return itemOut

def plotScatterplot2D(loadings,pdf,dat,nloads=3):
    """
    Plots Scatterplots 2D for a number of loadngs for PCA.

    :Arguments:
        :type loadings: pandas.DataFrame
        :param loadings: Loadings of the PCA.

        :type pdf: pdf object
        :param pdf: PDF object to save all the generated figures.

        :type dat: interface.wideToDesign
        :param dat: wideToDesign instance providing acces to all input data.

        :type nloads: int
        :param nloads: Number of principal components to create pairwise combs.
    """
    # Selecting amount of pairwise combinations to plot scaterplots for loads.
    for x, y in list(combinations(loadings.columns.tolist()[:nloads],2)):

        # Create a single-figure figure handler object
        fh = figureHandler(proj="2d", figsize=(14,8))

        # Create a title for the figure
        title = "{0} vs {1}".format(x,y)

        # Get colors
        if  dat.group:
            colorDes,ucGroups,combName=palette.getColors(design=dat.design,
                                                        groups=[dat.group])
            colorList = colorDes.colors.tolist()
        else:
            colorList = list(palette.mpl_colors[0]) #First color on the palette
            ucGroups = dict()

        # Plot the scatterplot based on data
        scatter.scatter2D(x=list(loadings[x]), y=list(loadings[y]),
                         colorList=colorList, ax=fh.ax[0])

        # Create legend
        fh.makeLegend(ax=fh.ax[0],ucGroups=ucGroups,group=dat.group)

        # Shrink axis to fit legend
        fh.shrink()

        # Despine axis
        fh.despine(fh.ax[0])

        # Formatting axis
        fh.formatAxis(figTitle=title,xTitle="Loadings on {0}".format(x),
            yTitle="Loadings on {0}".format(y),grid=False)

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
    fh = figureHandler(proj="3d")

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
    dat = wideToDesign(args.input, args.design, args.uniqID,group=args.group)

    #Dropping missing values
    if np.isnan(dat.wide.values).any():
        dat.wide = dropMissing(dat.wide)
    
    # RunPCA
    loadings,scores,block = runPCA(dat.wide)

    # Creatting outputs
    loadOut=createOutput(loadings,dat.wide.columns)
    scoreOut=createOutput(scores,dat.wide.index)  
    
    # Save loads
    with open(os.path.abspath(args.lname),"w") as LOADS:
        block.to_csv(LOADS,sep="\t", header=False)
        loadOut.to_csv(LOADS,sep="\t", index_label='sampleID')
    
    # Save scores
    with open(os.path.abspath(args.sname),"w") as SCORES:
        block.to_csv(SCORES,sep="\t", header=False)
        scoreOut.to_csv(SCORES,sep="\t", index_label=dat.uniqID)

    #Plotting scatter plot 3D
    logger.info(u"Plotting PCA scores")
    with PdfPages(os.path.abspath(args.figure)) as pdfOut:
        plotScatterplot2D(loadings=loadOut, pdf=pdfOut, dat=dat)
        plotScatterplot3D(data=loadOut, pdf=pdfOut, dat=dat)

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
