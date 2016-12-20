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
    standard.add_argument("-t", "--toCompare",dest="toCompare", action='store', 
                        required=True, default=True, help="Name of"
                        " the elements to compare in group col.")
    standard.add_argument("-n", "--nComp",dest="nComp", action='store', 
                        required=False,  default=3, type = int, help="Number"\
                        " of components.")

    output = parser.add_argument_group(title='Required output')
    output.add_argument("-os","--outScores",dest="outScores",action='store',required=True, 
                        help="Name of output file to store loadings. TSV format.")
    output.add_argument("-ow","--outWeights",dest="outWeights",action='store',required=True,
                        help="Name of output file to store weights. TSV format.")
    output.add_argument("-f","--figure",dest="figure",action="store",required=False,
                        help="Name of output file to store scatter plots for scores")

    development = parser.add_argument_group(title='Development Settings')
    development.add_argument("--debug",dest="debug",action='store_true',
                        required=False,help="Add debugging log output.")
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
    LVNames = ["LV_{0}".format(i) for i in range(nComp)]

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

def plotScatterplot(data,fh,x,y,design,title,group=False):
    """
    Creates an scater plot for 2 given components. It returns the ax with
    the figure.
    :Arguments:
        :type data: pandas.core.frame.DataFrame
        :param data: Data frame with the data to plot.

        :type ax: ax:matplotlib.axes._subplots.AxesSubplot.
        :param ax: empty axes.

        :type x: string
        :param x: Name of the first set of data.

        :type y: string
        :param y: name of the second set of data.

        :type design: interface.wideToDesign object.
        :param design: All standard data parsed

        :type title: string
        :param title: title for axis

        :type group: string
        :param group: Name of the column that contains the group information on the design file.

    :Return:
        :rtype fh: figureHandler.
        :returns ax: figureHandler containing ax with the scatter plot and legend.

    """
    logger.info(u"Creating scatter plot for components {0} vs {1} ".
                format(x,y))
    if group:
        glist=list(design[group])
        ch = colorHandler("tableau","Tableau_20")
        colorList, ucGroups = ch.getColorsByGroup(design=design,group=group,uGroup=sorted(set(glist)))
    else:
        glist = list()
        colorList = list('b') #blue
        ucGroups = dict()
    
    scatter.scatter2D(ax=fh.ax[0],x=list(data[x]),y=list(data[y]),colorList=colorList)

    fh.despine(fh.ax[0])

    if group:
        fh.makeLegend(ax=fh.ax[0],ucGroups=ucGroups,group=group)
        fh.shrink()

    #Get Variance
    fh.formatAxis(figTitle=title,xTitle="Scores on {0}".format(x),
        yTitle="Scores on {0}".format(y),grid=False)

    return fh

def plotScores(scores,pdf,dat):
    """
    This function creates a PDF file with 3 scatter plots for the combinations 
    of the 3 principal components. PC1 vs PC2, PC1 vs PC3, PC2 vs PC3.

    :Arguments:
        :type dat: interface.wideToDesign object.
        :param dat: All standard data parsed

        :type data: pandas.core.frame.DataFrame
        :param data: Data frame with the data to plot.
        
        :type PC1: string
        :param PC1:Name of the first set of data.

        :type PC2: string
        :param PC2:Name of the second set of data.

        :type PC3: string
        :param PC3:Name of the third set of data.

        :type outpath: string
        :param outpath: Path for the output file

        :type group: string
        :param group: Name of the column that contains the group information on the design file.

    :Return:
        :rtype PDF: file
        :retrn PDF: file with the 3 scatter plots for PC1 vs PC2, PC1 vs PC3, PC2  vs PC3.
    """
    for LV1,LV2 in  list(itertools.combinations(scores.columns.tolist(),2)):
        # Creating a figure handler object
        sFig = figureHandler(proj="2d", figsize=(14,8))

        # Creating title for the figure
        title = "{0} vs {1}".format(LV1,LV2)

        # Plotting
        fh1 = plotScatterplot(scores,sFig,LV1,LV2,dat.design.loc[scores.index,:],
                            title,dat.group)

        # Adding figure to pdf
        fh1.addToPdf(dpi=90,pdfPages=pdf)

def main(args):
    # Splitting groups to compare
    toCompare = args.toCompare.split(",")

    # Loading data trought Interface
    dat = wideToDesign(args.input, args.design, args.uniqID,group=args.group)

    # Treat everything as numeric
    dat.wide = dat.wide.applymap(float)

    #Dropping missing values
    if np.isnan(dat.wide.values).any():
        dat.wide = dropMissing(dat.wide)

    # Transpose data
    dat.trans = dat.transpose()

    #Run PLS
    scores_df,weights_df = runPLS(dat.trans,dat.group,toCompare,args.nComp)

    # Save scores_df
    scores_df.to_csv(args.outScores,sep="\t", index_label='sampleID')
    
    # Save weights_df
    weights_df.to_csv(args.outWeights,sep="\t", index_label=dat.uniqID)

    # Plotting scatter plot for scores
    with PdfPages(args.figure) as pdfOut:
        logger.info(u"Plotting PLS scores")
        plotScores(scores=scores_df,pdf=pdfOut,dat=dat)

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
    main(args)
