######################################################################################
# Date: 2016/July/11
# 
# Module: module_bar.py
#
# VERSION: 0.9
# 
# AUTHOR: Matt Thoburn (mthoburn@ufl.edu)
#
# DESCRIPTION: This module contains methods to plot a bar graph in matplotlib
#
#######################################################################################
import matplotlib.pyplot as plt
from manager_color import colorHandler
import pandas as pd
import numpy as np

def quickBar(ax,x,y):
    """
    This function draws a vertical bar graph

    :Arguments:
        :type ax: matplotlib Axis2D
        :param ax: Axis on which bar graph will be drawn

        :type x: list of numbers
        :param x: data to be plotted

        :type y: int
        :param y: dictates height of y axis 

    :Return:
        :type ax: Matplotlib Axis
        :param ax: axis with bars graphed onto it
    """
    width = 1/float(len(x)) + .5
  
    ticks = np.arange(len(x)) + 1

    ax.bar(left = ticks,height=y,width=width,color='b',align='center')
    ax.set_xticks(ticks)
    ax.set_xticklabels(x,rotation='vertical')
    return ax
def drawBars(ax,data,colors,dat=False):
    """
    This function draws a vertical bar graph

    :Arguments:
        :type ax: matplotlib Axis2D
        :param ax: Axis on which bar graph will be drawn

        :type data: list of numbers
        :param data: data to be plotted

        :type dat: pandas dataframe:
        :param dat: gets unique group list

        :type colors: list of colors
        :param colors: colors for bars with respect to a secondary group

    :Return:
        :type ax: Matplotlib Axis
        :param ax: axis with bars graphed onto it
    """ 
    if dat:
        numGroups = len(dat.levels)
    width = 1/float(len(data)) - .1
    ticks = np.arange(numGroups) + 1
    
    if len(colors) == 1:
        ax.bar(left = ticks,height=data,width=width,color=colors[0],align='center')
    else:   
        for i in range(len(data)):        
            barStart = [x+(i*width) for x in ticks]
            ax.bar(left=barStart,height=data[i],width=width,color=colors[i],align='center')

        #center ticks in middle of bar clusters
        ax.set_xticks((np.arange(numGroups)+1) + ((width/2) * (numGroups)))
    ax.set_xticklabels(dat.levels) #name ticks with group names
    
    return ax
#DEPREICATED vvv
def drawBar(ax,groups,ch,dat,field):
    """
    This function draws a vertical bar graph

    :Arguments:
        :type ax: matplotlib Axis2D
        :param ax: Axis on which bar graph will be drawn

        :type groups: list of strings
        :param groups: list of groups for optional color coding

        :type ch: colorHandler
        :param ch: colorHandler which determines how points are colored with respect to group

        :type dat: pandas dataframe:
        :param dat: data for graphing

        :type field: string:
        :param field: name of column to be graphed

    :Return:
        :type ax: Matplotlib Axis
        :param ax: axis with bars graphed onto it
    """ 
    groupsNoDupes = vm.noDupes(groups)
    xAxLen = len(vm.noDupes(groups))
    values = list(dat[field])
    if len(ch.hexClrs) == 1:
        print np.arange(xAxLen)
        print groups
        ax.bar(left=np.arange(xAxLen),height=values,width=.3,color=ch.hexClrs[0],align='center',tick_label=groups)
    else:
        clrSz = len(ch.groupsNoDupes)
        groupList2D = np.zeros((clrSz,xAxLen))
        #print type(groupList2D)
        for i in range(0,len(dat[field])):

            grpIndex = vm.indexOf(ch.groupsNoDupes,ch.groupList[i])
            otherGrpIndex = vm.indexOf(groupsNoDupes,groups[i])
            groupList2D[grpIndex][otherGrpIndex] = dat[field][i]
        for i in range(0,clrSz):
            ticks = np.arange(xAxLen)
            barStart = [x+(i*.3) for x in ticks]

            ax.bar(left=barStart,height=groupList2D[i],width=.3,color=ch.hexClrs[i])        
    return ax
