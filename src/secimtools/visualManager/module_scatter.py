######################################################################################
# Date: 2016/July/11
#
# Module: module_2DScatter.py
#
# VERSION: 0.9
#
# AUTHOR: Matt Thoburn (mthoburn@ufl.edu); 
#         edited by Miguel A. Ibarra-Arellano(miguelib@ufl.edu)
#
# DESCRIPTION: This module contains a primary method (quickPlot) 
#              and two depreciated methods which are not to be used
#
#              makeScatter is to be called from other scripts which require a graph
#######################################################################################
def scatter2D(ax,x,y,colorList,ec='black'):
    """
    This function is to be called by makeScatter2D, creates a 2D scatter plot on a given axis with
    colors determined by the given colorHandler or an optional override
    :Arguments:

        :type ax: matplotlib Axis2D
        :param ax: Axis on which scatter plot will be drawn

        :type x: list of floats
        :param x: list of x values to be plotted

        :type y: list of floats
        :param y: list of y values to be plotted

        :type colorList: list 
        :param colorList: list of colors to be used for plotting

        :type ec: str 
        :param ec: Edge color for markers

    :Return:
        :type ax: Matplotlib Axis
        :param ax: axis with scatter plotted onto it

    """ 
    ax.scatter(x,y,color=colorList,marker='o',s=50,edgecolors=ec, rasterized=True)
    
    return ax
def scatter3D(ax,x,y,z,colorList):
    """
    This function is to be called by makeScatter3D, creates a 3D scatter plot on a given axis with
    colors determined by the given colorHandler. 

    :Arguments:
        :type ax: matplotlib Axis3D
        :param ax: Axis on which scatter plot will be drawn

        :type x: list of floats
        :param x: list of x values to be plotted

        :type y: list of floats
        :param y: list of y values to be plotted

        :type z: list of floats
        :param z: list of z values to be plotted

        :type colorList: list
        :param colorList: list of colors to be used for plotting

    
    :Return:
        :type ax: Matplotlib Axis
        :param ax: axis with scatter plotted onto it
    """
    ax.scatter(xs=x,ys=y,zs=z,c=colorList,marker='o',s=50,depthshade=False, rasterized=True)

    
    return ax
            
