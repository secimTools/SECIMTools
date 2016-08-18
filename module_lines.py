######################################################################################
# Date: 2016/July/11
# 
# Module: module_lines.py
#
# VERSION: 0.9
# 
# AUTHOR: Matt Thoburn (mthoburn@ufl.edu)
#
# DESCRIPTION: This module contains several methods for drawing cutoff lines
#
#              These will usually be drawn on top of a scatter plot or other graph
#######################################################################################
from matplotlib.lines import Line2D
#x and y can be a single value or a list of values
def drawCutoffVert(ax,x,cl="r",lb="",ls="--",lw=.5):
    """
    This method draws a vertical cutoff line or lines
    :Arguments:

        :type ax: matplotlib axis
        :param ax: axis to draw lines on

        :type x: int/float or list of ints/floats
        :param x: x value(s) on which to draw cutoff lines

        :type cl: string
        :param cl: color. default is red

        :type lb: string
        :param lb: optional label for legend

        :type ls: string
        :param ls: optional linestyle defaults to solid line

        :type lw: int
        :param lw: optional line width

    :Return:
        :type ax: matplotlib axis
        :param ax: axis with cutoffs
    """
    oldLims = ax.get_ylim()
    ax.vlines(x,ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1],color='r',label=lb,linestyle=ls,linewidth=lw)
    ax.set_ylim(oldLims)
    return ax
    
def drawCutoffHoriz(ax,y,cl="r",lb="",ls="--",lw=.5):
    """
    This method draws a vertical cutoff line or lines
    :Arguments:

        :type ax: matplotlib axis
        :param ax: axis to draw lines on

        :type y: int/float or list of ints/floats
        :param y: y value(s) on which to draw cutoff lines

        :type cl: string
        :param cl: color. default is red

        :type lb: string
        :param lb: optional label for legend

        :type ls: string
        :param ls: optional linestyle defaults to solid line

        :type lw: int
        :param lw: optional line width

    :Return:
        :type ax: matplotlib axis
        :param ax: axis with cutoffs
    """

    #get old limits
    limits = ax.get_xlim()

    #print line
    ax.hlines(y,xmin=limits[0],xmax=limits[1],color=cl,label=lb,linestyle=ls,
                linewidth=lw)

    #Returning
    return ax
    
def drawCutoff(ax,x,y,c='r'):
    """
    This method draws a vertical cutoff line or lines
    :Arguments:

        :type ax: matplotlib axis
        :param ax: axis to draw lines on

        :type x: int/float or list of ints/floats
        :param x: x value(s) on which to draw cutoff lines

        :type y: int/float or list of ints/floats
        :param y: y value(s) on which to draw cutoff lines

        :type c: string
        :param c: color. default is red
        
    :Return:
        :type ax: matplotlib axis
        :param ax: axis with cutoffs
    """
    line = Line2D(x,y,color=c)
    ax.add_line(line)
    return ax
    
