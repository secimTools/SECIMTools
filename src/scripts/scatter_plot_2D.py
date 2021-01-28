#!/usr/bin/env python
################################################################################
# DATE: 2016/August/10
#
# MODULE: scatter_plot_2D.py
#
# VERSION: 1.1
#
# AUTHOR: Matt Thoburn (mthoburn@ufl.edu), Miguel Ibarra (miguelib@ufl.edu)
#
# DESCRIPTION: This is a standalone executable for graphing 2D scatter plots 
# from wide data
################################################################################
# Import built-in libraries
import os
import logging
import argparse
from argparse import RawDescriptionHelpFormatter
# Import add-on libraries
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
# Import local data libraries
from secimtools.dataManager import logger as sl
from secimtools.dataManager.interface import wideToDesign
# Import local plotting libraries
from secimtools.visualManager import module_scatter as scatter
from secimtools.visualManager.manager_color import colorHandler
from secimtools.visualManager.manager_figure import figureHandler


def getOptions():
    description="Stand alone Scatter Plots 2D tool"
    parser = argparse.ArgumentParser(description=description, 
                        formatter_class=RawDescriptionHelpFormatter)
    # Standard input
    standard = parser.add_argument_group(title="Standard input", 
                        description='Standard input for SECIM tools.')
    standard.add_argument( "-i","--input", dest="input", action='store', 
                        required=True, help="Input dataset in wide format.")
    standard.add_argument("-d" ,"--design",dest="design", action='store', 
                        required=False,default=False, help="Design file.")
    standard.add_argument("-id", "--ID",dest="uniqID", action='store', 
                        required=True, help="Name of the column with unique "\
                        "identifiers.")
    standard.add_argument("-g", "--group",dest="group", action='store', 
                        required=False, default=False, help="Name of the column"\
                        " with groups.")
    # Tool input
    tool = parser.add_argument_group(title="Tool especific input")
    tool.add_argument("-x" ,"--X",dest="x", action='store', 
                        required=True, help="Name of column for X values")
    tool.add_argument("-y", "--Y",dest="y", action='store', 
                        required=True, help="Name of column for Y values")
    # Tool output
    output = parser.add_argument_group(title="Output")
    output.add_argument("-f","--figure",dest="figure",action="store",
                        required=False, help="Path of figure.")
    # Plot options
    plot = parser.add_argument_group(title='Plot options')
    plot.add_argument("-pal","--palette",dest="palette",action='store',required=False, 
                        default="tableau", help="Name of the palette to use.")
    plot.add_argument("-col","--color",dest="color",action="store",required=False, 
                        default="Tableau_20", help="Name of a valid color scheme"\
                        " on the selected palette")
    args = parser.parse_args()

    # Standardize paths
    args.input  = os.path.abspath(args.input)
    args.figure = os.path.abspath(args.figure)
    if args.design:
        args.design = os.path.abspath(args.design)
    return(args)


def main(args):
    # Loading design
    if args.design:
    #    design = pd.DataFrame.from_csv(args.design,sep="\t")
        design = pd.read_csv(args.design,sep="\t")
    design.reset_index(inplace=True)
#    else:
#        design = False

    # Loading wide file
   #KAS: from_csv is deprecated
   # wide = pd.DataFrame.from_csv(args.input,sep="\t")
    wide = pd.read_csv(args.input,sep="\t")

    # Create figureHandler object
    fh = figureHandler(proj="2d",figsize=(14,8))

    # If design file with group and the uniqID is "sampleID" then color by group
    if args.group and args.uniqID == "sampleID":
        glist=list(design[args.group])
        colorList, ucGroups = palette.getColorsByGroup(design=design,
                                group=args.group,uGroup=sorted(set(glist)))
    else:
        glist = list()
        colorList = palette.mpl_colors[0]
        ucGroups = dict()

    # Plote scatterplot 2D
    scatter.scatter2D(ax=fh.ax[0], x=list(wide[args.x]), y=list(wide[args.y]),
                    colorList=colorList)

    # Despine axis (spine = tick)
    fh.despine(fh.ax[0])

    # Formating axis
    fh.formatAxis(figTitle=args.x + " vs " + args.y, xTitle=args.x, 
                yTitle=args.y, grid=False)

    # If groups are provided create a legend
    if args.group and args.uniqID == "sampleID":
        fh.makeLegend(ax=fh.ax[0],ucGroups=ucGroups,group=args.group)
        fh.shrink()

    # Saving figure to file
    with PdfPages(args.figure) as pdfOut:
        fh.addToPdf(dpi=600, pdfPages=pdfOut)
    logger.info("Script Complete!")


if __name__ == '__main__':
    args   = getOptions()
    logger = logging.getLogger()
    sl.setLogger(logger)
    palette = colorHandler(pal=args.palette, col=args.color)
    logger.info(u"Using {0} color scheme from {1} palette".format(args.color,
                args.palette))
    main(args)
