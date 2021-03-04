#!/usr/bin/env python
################################################################################
#
# SCRIPT: coefficient_vairation_flags.py
# The output is a pdf of distributions, and a spreadsheet of flags
#
# AUTHORS: Miguel A Ibarra <miguelib@ufl.edu>
#          Matt Thoburn <mthoburn@ufl.edu>
#          Oleksandr Moskalenko <om@rc.ufl.edu>
#
################################################################################

import os
import logging
import argparse
from math import log, floor
from argparse import RawDescriptionHelpFormatter

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from secimtools.dataManager import logger as sl
from secimtools.dataManager.flags import Flags
from secimtools.dataManager.interface import wideToDesign

from secimtools.visualManager import module_bar as bar
from secimtools.visualManager import module_box as box
from secimtools.visualManager import module_hist as hist
from secimtools.visualManager import module_lines as lines
from secimtools.visualManager import module_scatter as scatter
from secimtools.visualManager import module_distribution as dist
from secimtools.visualManager.manager_color import colorHandler
from secimtools.visualManager.manager_figure import figureHandler


def getOptions(myopts=None):
    """Function to pull in arguments"""
    description="""The coefficient of variation (CV) is defined as the ratio
    of the sample standard deviation to the mean. It is a method to measure the
    variations of compounds. The variation of a peak intensity increases as
    its CV increases. And adjusted for the sample mean, CV does not have unit;
    thus, it is a standardized measurement for variation.

    A density plot of CVs for all compounds across samples by group is performed.
    And a set of flags of compounds with large CVs will be output. """

    parser=argparse.ArgumentParser(description=description,
                                formatter_class=RawDescriptionHelpFormatter)
    # Standard Input
    standard = parser.add_argument_group(title="Standard input",
                                description="Standard input for SECIM tools.")
    standard.add_argument("-i","--input", dest="input", action="store",
                        required=True, help="Input dataset in wide format.")
    standard.add_argument("-d","--design", dest="design", action="store",
                        required=True, help="Design file.")
    standard.add_argument("-id","--ID", dest="uniqID", action="store", required=True,
                         help="Name of the column with unique identifiers.")
    standard.add_argument("-g","--group", dest="group", action='store',
                        required=False, default=False, help="Name of column in "\
                        "design file with Group/treatment information.")
    standard.add_argument("-l","--levels",dest="levels",action="store",
                        required=False, default=False, help="Different groups to"\
                        "sort by separeted by commas.")
    # Tool Input
    tool = parser.add_argument_group(title="Tool specific input",
                                description="Input specific for the tool.")
    tool.add_argument("-c","--CVcutoff", dest="CVcutoff", action="store",
                        required=False, default=False, type=float,
                        help="The default CV cutoff will flag 10 percent of "\
                        "the rowIDs with larger CVs. If you want to set a CV "\
                        "cutoff, put the number here. [optional]")
    # Tool output
    output = parser.add_argument_group(title="Output",
                            description="Paths for output files.")
    output.add_argument("-f","--figure", dest="figure", action="store",
                        required=True, default="figure",
                        help="Name of the output PDF for CV plots.")
    output.add_argument("-o","--flag", dest="flag", action="store",
                        required=True, default="RTflag",
                        help="Name of the output TSV for CV flags.")
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
    args.design = os.path.abspath(args.design)
    args.figure = os.path.abspath(args.figure)
    args.flag   = os.path.abspath(args.flag)

    # if args.levels then split otherwise args.level = emptylist
    if args.levels:
        args.levels = args.levels.split(",")

    return(args)


def calculateCV(data, design, cutoff, levels):
    """
    Runs Count Values by group

    :Arguments:
        :type data: pandas.DataFrame
        :param data: wide dataset

        :type design: pandas.DataFrame
        :param design: design dataset

        :type cutoff: float
        :param cutoff: Value of cutoff if non provided it will be calculated.

        :type levels: str
        :param levels: Name of the groups to groupby
    # Get max CV
    """

    #Open CV and CVcutoffs dataframes
    CV = pd.DataFrame(index=data.index)
    CVcutoff = pd.Series(index=list(set(design[levels])), dtype="Float64")

    # Split design file by treatment group
    for title, group in design.groupby(levels):
        # Create empty dataset with metabolites names as index and calculate their
        # standar deviation and mean
        DATstat=pd.DataFrame(index=data[group.index].index)
        # ddof =1 is necessary to subtract n-1 in denominatior for standard deviation.
        DATstat.loc[:,"std"]  = np.std(data[group.index], axis=1, ddof=1)
        DATstat.loc[:,"mean"] = np.mean(data[group.index],axis=1)

        # Calculate the Coefficient of Variation for that group (if groups)
        # or al data (if not groups provided).
        CV.loc[:,"cv_"+title] = abs(DATstat["std"] / DATstat["mean"])

        # Calculate the CVcutoffs for each group (if groups provided)
        # or all data (if not).
        if not cutoff:
            CVcutoff[title] = np.nanpercentile(CV["cv_"+title].values, q=90)
            CVcutoff[title] = round(CVcutoff[title],
                                -int(floor(log(abs(CVcutoff[title]),10)))+2)
        else:
            CVcutoff[title] = np.nanpercentile(CV["cv_"+title].values, q=(1-cutoff)*100 )
            CVcutoff[title] = round(CVcutoff[title],
                                -int(floor(log(abs(CVcutoff[title]),10)))+2)

    # Calculate the maximum coefficient of variation
    CV.loc[:,'cv'] = CV.apply(np.max, axis=1)

    return (CV, CVcutoff)


def plotCVplots(data, cutoff, palette, pdf):
    #Iterate over groups
    for name,group in palette.design.groupby(palette.combName):
        # Open figure handler
        fh=figureHandler(proj='2d',figsize=(14,8))

        # Get xmin and xmax
        xmin = -np.nanpercentile(data['cv_'+name].values,99)*0.2
        xmax = np.nanpercentile(data['cv_'+name].values,99)*1.5

        # Plot histogram
        hist.serHist(ax=fh.ax[0],dat=data['cv_'+name],color='grey',normed=1,
                    range=(xmin,xmax),bins=15)

        # Plot density plot
        dist.plotDensityDF(data=data['cv_'+name],ax=fh.ax[0], lb="CV density",
                           colors=palette.ugColors[name])

        # Plot cutoff
        lines.drawCutoffVert(ax=fh.ax[0],x=cutoff[name], lb="Cutoff at: {0}".format(cutoff[name]))

        # Plot legend
        fh.makeLegendLabel(ax=fh.ax[0])

        # Give format to the axis
        fh.formatAxis(yTitle='Density',xlim=(xmin,xmax), ylim="ignore",
            figTitle = "Density Plot of Coefficients of Variation in {0}".format(name))

        # Shrink figure to fit legend
        fh.shrink()

        # Add plot to PDF
        fh.addToPdf(pdfPages=pdf)


def plotDistributions(data, cutoff, palette,pdf):
    # Open new figureHandler instance
    fh=figureHandler(proj='2d', figsize=(14,8))

    #Get xmin and xmax
    xmin = -np.nanpercentile(data['cv'].values,99)*0.2
    xmax = np.nanpercentile(data['cv'].values,99)*1.5


    # Split design file by treatment group and plot density plot
    for name, group in palette.design.groupby(palette.combName):
        dist.plotDensityDF(data=data["cv_"+name],ax=fh.ax[0],colors=palette.ugColors[name],
                            lb="{0}".format(name))

    # Plot legend
    fh.makeLegendLabel(ax=fh.ax[0])

    # Give format to the axis
    fh.formatAxis(yTitle="Density", xlim=(xmin,xmax), ylim="ignore",
        figTitle="Density Plot of Coefficients of Variation by {0}".format(palette.combName))

    fh.shrink()
    fh.addToPdf(pdfPages=pdf)


def main(args):
    """ Function to input all the arguments"""
    # Checking if levels
    if args.levels and args.group:
        levels = [args.group]+args.levels
    elif args.group and not args.levels:
        levels = [args.group]
    else:
        levels = []
    logger.info("Groups used to color by: {0}".format(",".join(levels)))

    # Import data
    dat = wideToDesign(args.input, args.design, args.uniqID, group=args.group,
                    anno=args.levels, logger=logger)

    # Remove groups with just one element
    dat.removeSingle()

    # Cleaning from missing data
    dat.dropMissing()

    # Treat everything as float and round it to 3 digits
    dat.wide = dat.wide.applymap(lambda x: round(x,3))

    # Get colors
    palette.getColors(dat.design,levels)

    # Use group separation or not depending on user input

    CV,CVcutoff = calculateCV(data=dat.wide, design=palette.design,
                              cutoff=args.CVcutoff, levels=palette.combName)

    # Plot CVplots for each group and a distribution plot for all groups together
    logger.info("Plotting Data")
    with PdfPages(args.figure) as pdf:
        plotCVplots      (data=CV, cutoff=CVcutoff, palette=palette, pdf=pdf)
        plotDistributions(data=CV, cutoff=CVcutoff, palette=palette, pdf=pdf)

    # Create flag file instance and output flags by group
    logger.info("Creatting Flags")
    flag = Flags(index=CV['cv'].index)
    for name, group in palette.design.groupby(palette.combName):
        flag.addColumn(column="flag_feature_big_CV_{0}".format(name),
                       mask=((CV['cv_'+name] > CVcutoff[name]) | CV['cv_'+name].isnull()))

    flag.df_flags.to_csv(args.flag, sep='\t')

    logger.info("Script Complete!")


if __name__ == '__main__':
    args=getOptions(myopts=True)
    logger=logging.getLogger()
    sl.setLogger(logger)
    logger.info("Importing data with following parameters: "\
                "\n\tWide: {0}"\
                "\n\tDesign: {1}"\
                "\n\tUnique ID: {2}"\
                "\n\tGroup: {3}".format(args.input, args.design, args.uniqID,
                    args.group))
    # Etablish a color palette for data and cutoffs
    palette = colorHandler(pal=args.palette, col=args.color)
    logger.info("Using {0} color scheme from {1} palette".format(args.color,
                args.palette))
    main(args)
