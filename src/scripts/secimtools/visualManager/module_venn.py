#!/usr/bin/env python
################################################################################
# DATE: 2016/Jun/01
#
# SCRIPT: module_venn.py
#
# VERSION: 1.0
# 
# AUTHOR: Miguel A Ibarra (miguelib@ufl.edu)
# 
# DESCRIPTION: This script creates a venn diagrame for either 2 or 3 groups.
#               It contains the following packages
################################################################################

#Standar Libraries

#AddOn Libraries
from matplotlib import pyplot as plt
from matplotlib_venn import venn2, venn2_circles
from matplotlib_venn import venn3, venn3_circles
from matplotlib.backends.backend_pdf import PdfPages

# Local Packages
from secimtools.visualManager.manager_color import colorHandler
from secimtools.visualManager.manager_figure import figureHandler


def plotVenn2(data,title,name1,name2,innerLabels=None,circles=None):
    """ 
    Plots venn diagram for 2 sets (2 circles).

    :Arguments:

        :type data: list
        :param data: list of values for venn circles [1,2,3] = [Ab,AB,aB]
        
        :type title: str
        :param title: Title for the plot
        
        :type name1: str
        :param name1: Name of the first category (circle)
        
        :type name2: str
        :param name2: Name of the second category (circle)
        
        :type innerLabels: list
        :param innerLabels: List of labels for the inside of circles.
        
        :type circles: boolean
        :param circles: If true draws the edge of the circles


    :Returns:
        :rtype figInstance: figureHandler object
        :returns figInstance: Outputs a figureHandler object with the plot.

    """

    #Get figure instances
    figInstance=figureHandler(proj="2d")

    #Setting format of the figure
    figInstance.formatAxis(xTitle=name1,yTitle=name2,axTitle=title)

    #Plotting venn
    venn2fig = venn2(subsets=data, set_labels=(name1,name2), ax=figInstance.ax[0])

    #Plot circles
    if circles:
        circles = venn2_circles(subsets=data, linestyle='dotted',ax=figInstance.ax[0])

    # If not inner labels are provided use the data as a string
    if innerLabels is None:
        innerLabels=list(map(str,data))

    #Art of the venn diagram
    _artVenn2(venn2fig,innerLabels=innerLabels)

    #Return Plot
    return figInstance

def _artVenn2(venn,innerLabels):
    """ 
    Modifies the colors, text and alpha for the 2 cicles venn plot.

    :Arguments:

        :type data: matplotlib.figure
        :param data: Figure with the venn diagram
        
        :type innerLabels: list
        :param innerLabels: List of the labels to be printed inside the circles
                            [Ab,AB,aB]
    """

    # Subset innerLabels
    venn.get_label_by_id('10').set_text(innerLabels[0])
    venn.get_label_by_id('01').set_text(innerLabels[1])
    venn.get_label_by_id('11').set_text(innerLabels[2])

    #Selecting colours from colour manager
    palette = colorHandler(pal="tableau",col="BlueRed_6")

    # Subset colors
    venn.get_patch_by_id('10').set_color(palette.mpl_colors[3])
    venn.get_patch_by_id('01').set_color(palette.mpl_colors[4])
    venn.get_patch_by_id('11').set_color(palette.mpl_colors[5])

    # Subset alphas
    venn.get_patch_by_id('10').set_alpha(1.0)
    venn.get_patch_by_id('01').set_alpha(1.0)
    venn.get_patch_by_id('11').set_alpha(1.0)


###########
# TO DO 
###########
#