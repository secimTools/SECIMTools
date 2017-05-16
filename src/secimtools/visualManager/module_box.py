######################################################################################
# Date: 2016/July/11
# 
# Module: module_box.py
#
# VERSION: 0.9
# 
# AUTHOR: Miguel Ibarra (miguelib@ufl.edu), Matt Thoburn (mthoburn@ufl.edu)
#
# DESCRIPTION: This module contains a method to plot a box and whisker plot in matplotlib
#
#######################################################################################
import pandas as pd

def boxDF(dat,ax,colors,vert=True,rot=90):
    """
    This function draws a vertical box and whisker plot
    :Arguments:
    
        :type ax: matplotlib Axis2D
        :param ax: Axis on which bar graph will be drawn

        :type colors: list
        :param colors: list of colors per sample, they should match the order
                        of the sample

        :type dat: pandas dataframe:
        :param dat: data for graphing

    :Return:
        :type ax: Matplotlib Axis
        :param ax: axis with bars graphed onto it
    """
    bp = dat.boxplot(ax=ax, sym='b+', rot=rot, showmeans=True, meanline=False,
                    patch_artist=True,vert=vert, return_type="dict", grid=False)

    #Set art on boxes
    for color,box in zip(colors,bp['boxes']):
        #print color
        box.set(color='k')
        box.set_alpha(0.7)
        box.set_facecolor(color)

    #Set art on whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='k')

    #Set art on fliers
    for flier in bp['fliers']:
        flier.set_markersize(7)
        flier.set_marker('o')
        flier.set(marker='o',color='k')

    #Set colors for caps
    for cap in bp['caps']:
        cap.set(color="k")

    #Set art on medians
    for median in bp['medians']:
        median.set(color="b")
        median.set(marker="_")
        median.set(linewidth=2)

    #Set art on means
    for mean in bp['means']:
        mean.set(color="r")
        mean.set(marker="s")

    return ax

def boxSeries(ax,ser):
    """
    This function draws a vertical box and whisker plot

    :Arguments:
        :type ax: matplotlib Axis2D
        :param ax: Axis on which bar graph will be drawn

        :type ser: pandas Series
        :param ser: data to be plotted

    :Return:
        :type ax: Matplotlib Axis
        :param ax: axis with bars graphed onto it
    """
    bp = ax.boxplot(ser,showfliers=True)
    for flier in bp['fliers']:
        flier.set(marker='+', color='#FFFFFF')
    return ax
