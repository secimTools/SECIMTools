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
    standard.add_argument( "-i","--input", dest="input", action='store', required=True, 
                        help="Input dataset in wide format.")
    standard.add_argument("-d" ,"--design",dest="design", action='store', required=True,
                        help="Design file.")
    standard.add_argument("-id", "--ID",dest="uniqID", action='store', required=True, 
                        help="Name of the column with unique identifiers.")
    standard.add_argument("-g", "--group",dest="group", action='store', required=False, 
                        default=False,help="Name of the column with groups.")

    output = parser.add_argument_group(title='Required output')
    output.add_argument("-lo","--load_out",dest="lname",action='store',required=True, 
                        help="Name of output file to store loadings. TSV format.")
    output.add_argument("-so","--score_out",dest="sname",action='store',required=True,
                        help="Name of output file to store scores. TSV format.")
    output.add_argument("-f","--figure",dest="figure",action="store",required=False,
                        help="Name of output file to store scatter plots for 3 \
                        principal components.")

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
    loadings = pca.components_

    #Get aditional information out of PCA, this will mimic the output from R
    sd = scores.std(axis=0)
    propVar = pca.explained_variance_ratio_
    cumPropVar = propVar.cumsum()

    #Crete labels(index) for the infor
    labels = np.array(['#Std. deviation', '#Proportion of variance explained',
                        '#Cumulative proportion of variance explained'])

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

def plotScatterplot(data,fh,block,x,y,dat,title,group=False):
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

        :type dat: interface.wideToDesign object.
        :param dat: All standard data parsed

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
        glist=list(dat.design[group])
        ch = colorHandler("tableau","Tableau_20")
        colorList, ucGroups = ch.getColorsByGroup(design=dat.design,group=group,uGroup=sorted(set(glist)))
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
    varx=block.loc["#Proportion of variance explained",int(x.replace("PC",""))-1]
    vary=block.loc["#Proportion of variance explained",int(y.replace("PC",""))-1]
    fh.formatAxis(figTitle=title,xTitle=x+" {0:.2f}%".format(varx*100),
        yTitle=y+" {0:.2f}%".format(vary*100),grid=False)

    return fh

def plotPCA(dat,data,block,PC1,PC2,PC3,outpath,group=False):
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
    logger.info(u"Plotting principal components")
    #Creatig output PDF
    pdfOut = PdfPages(os.path.abspath(outpath))

    #Creating the figure
    title = "{0} vs {1}".format(PC1,PC2)
    fh1 = figureHandler(proj="2d")

    #Plotting
    fh1 = plotScatterplot(data,fh1,block,PC1,PC2,dat,title,group,)
    fh1.addToPdf(dpi=90,pdfPages=pdfOut)

    title = "{0} vs {1}".format(PC1,PC3)
    fh2 = figureHandler(proj="2d")

    fh2 = plotScatterplot(data,fh2,block,PC1,PC3,dat,title,group)
    fh2.addToPdf(dpi=90,pdfPages=pdfOut)

    title = "{0} vs {1}".format(PC2,PC3)
    fh3 = figureHandler(proj="2d")

    fh3 = plotScatterplot(data,fh3,block,PC2,PC3,dat,title,group)
    fh3.addToPdf(dpi=90,pdfPages=pdfOut)
    
    fh4 = figureHandler(proj="3d")
    if group:
        glist=list(dat.design[group])
        ch = colorHandler("tableau","Tableau_20")
        colorList, ucGroups = ch.getColorsByGroup(design=dat.design,group=group,uGroup=sorted(set(glist)))
    else:
        glist = list()
        colorList = list('b') #blue
        ucGroups = dict()
        
    ax = scatter.scatter3D(ax=fh4.ax[0],colorList=colorList,x=list(data[PC1]),y=list(data[PC2]),z=list(data[PC3]))
    if group:
        fh4.makeLegend(ax=fh4.ax[0],ucGroups=ucGroups,group=group)
    fh4.format3D(xTitle=PC1,yTitle=PC2,zTitle=PC3)
    fh4.addToPdf(dpi=90,pdfPages=pdfOut)

    pdfOut.close()

def main():
    # Command line options
    args = getOptions()

    #Set logger
    global logger
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

    #Loading data trought Interface
    dat = wideToDesign(args.input, args.design, args.uniqID,group=args.group)

    #Dropping missing values
    if np.isnan(dat.wide.values).any():
        dat.wide = dropMissing(dat.wide)

    #Run PCA
    loadings,scores,block = runPCA(dat.wide)
 
    #print loadings
    #Getting output tables
    loadOut=createOutput(loadings, dat.wide.columns)
    scoreOut=createOutput(scores,dat.wide.index)

    # Save output
    # Save loads
    with open(os.path.abspath(args.lname),"w") as LOADS:
        block.to_csv(LOADS,sep="\t", header=False)
        loadOut.to_csv(LOADS,sep="\t", index_label='sampleID')
    
    # Save scores
    with open(os.path.abspath(args.sname),"w") as SCORES:
        block.to_csv(SCORES,sep="\t", header=False)
        scoreOut.to_csv(SCORES,sep="\t", index_label=dat.uniqID)

    #Plotting scatter plot 3D
    plotPCA(dat,loadOut,block,"PC1","PC2","PC3",args.figure,args.group)

    #Ending script
    logger.info(u"Finishing running of PCA")


if __name__ == '__main__':
    main()
