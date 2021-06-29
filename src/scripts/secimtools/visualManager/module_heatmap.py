#!/usr/bin/env python
################################################################################
# DATE: 2016/May/30
#
# SCRIPT: hcheatmap.py
#
# VERSION: 2.0
# 
# AUTHOR: Miguel A Ibarra (miguelib@ufl.edu)
# 
# DESCRIPTION: This script takes a a wide format file and plots a heatmap of
#               the data. This tool is usefull to get a representation of the
#               data.
#
#     input     : Path to Wide formated file.
#     design    : Path to Design file corresponding to wide file.
#     id        : Uniq ID on wide file (ie. RowID).
#     heatmap   : Output path for loads heatmap.
# 
# The output is a heatmap that describes the wide dataset.
################################################################################

#AddOn Libraries
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# Local Packages
from secimtools.visualManager.manager_color import colorHandler


def plotHeatmap(data,ax,cmap=False,xlbls=True,ylbls=True):
    """
    This function creates and plots a heatmap or a hcheatmap (hiearchical
    cluster heat map)

    :Arguments:
        :type data: pandas data frame.
        :param data: Data that is going to be used to plot

        :type ax: matplotlib axis
        :param ax: axis to be drawn on.
    :Returns:
        :return ax: matplotlib axis.
        :rtype ax: axis with the drawn figure.
    """
    
    # Create a custom colormap for the heatmap values
    # You can have two possible ways to create a pallete.
    # 1) sns.diverging_palette (a seaborn function)
    # 2) palettable.colorbrewer.diverging.[color].mpl_colors 

    if not cmap:
        colors = colorHandler(pal="diverging", col="Spectral_10")
        cmap = colors.mpl_colormap

    # Draw the full plot
    sns.heatmap(data,linewidths=0.0,ax=ax,cmap=cmap,xticklabels=xlbls,
                yticklabels=ylbls)

def plotHCHeatmap(data,hcheatmap=True,cmap=False,xlbls=True,ylbls=True):
    """
    This function creates and plots a heatmap or a hcheatmap (hiearchical
    cluster heat map)

    :Arguments:
        :type data: pandas data frame.
        :param data: Data that is going to be used to plot

        :type hcheatmap: boolean
        :param hcheatmap: boolean to determinate wether you want to plot a normal heatmap or a hcheatmap.
    :Returns:
        :return hmap: Instance of the heatmap
        :rtype hmap: seaborn.matrix.ClusterGrid
    """
    
    # Create a custom colormap for the heatmap values
    # You can have two possible ways to create a pallete.
    # 1) sns.diverging_palette (a seaborn function)
    # 2) palettable.colorbrewer.diverging.[color].mpl_colors 
    if not cmap:
        colors = colorHandler(pal="diverging", col="Spectral_10")
        cmap = colors.mpl_colormap

    # Draw the full plot
    hmap = sns.clustermap(data,row_cluster=hcheatmap,col_cluster=hcheatmap,
                        linewidths=0.0,figsize=(13,13),cmap=cmap,
                        xticklabels=xlbls,yticklabels=ylbls)

    #Rotate axis
    plt.setp(hmap.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.setp(hmap.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)

    #Return Plot
    return hmap