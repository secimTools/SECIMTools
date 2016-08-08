#!/usr/bin/env python
################################################################################
# DATE: 2016/May/06, rev: 2016/June/03
#
# SCRIPT: flag.py
#
# VERSION: 1.1
# 
# AUTHOR: Miguel A Ibarra (miguelib@ufl.edu) Edited by: Matt Thoburn (mthoburn@ufl.edu)
# 
# The output is a pdf of distributions, and a spreadsheet of flags
#
################################################################################
# Built-in packages
import logging

import argparse
from argparse import RawDescriptionHelpFormatter
import tempfile
import shutil
import os
from math import log, floor

# Add-on packages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

# Local packages
import logger as sl
from interface import wideToDesign
from flags import Flags
from manager_color import colorHandler
from manager_figure import figureHandler
import module_lines as lines
import module_distribution as dist
import module_hist as hist

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

    group1=parser.add_argument_group(title="Standard input", 
                                description="Standard input for SECIM tools.")
    group1.add_argument("-i","--input", dest="input", action="store", 
                        required=True, help="Input dataset in wide format.")
    group1.add_argument("-d","--design", dest="design", action="store", 
                        required=True, help="Design file.")
    group1.add_argument("-id","--ID", dest="uniqID", action="store", required=True,
                         help="Name of the column with unique identifiers.")
    group1.add_argument("-f","--figure", dest="figure", action="store", 
                        required=True, default="figure", 
                        help="Name of the output PDF for CV plots.")
    group1.add_argument("-o","--flag", dest="flag", action="store", 
                        required=True, default="RTflag", 
                        help="Name of the output TSV for CV flags.")

    group2=parser.add_argument_group(title="Optional input", 
                            description="Optional input for SECIM tools.")
    group2.add_argument("-g","--group", dest="group", action="store", 
                        required=False, default=False, 
                        help="""Add option to separate sample IDs by treatment 
                        name. [optional]""")
    group2.add_argument("-c","--CVcutoff", dest="CVcutoff", action="store", 
                        required=False, default=False, 
                        help="""The default CV cutoff will flag 10 percent of 
                        the rowIDs with larger CVs. If you want to set a CV 
                        cutoff, put the number here. [optional]""")

    args=parser.parse_args()

    return(args)

def setflagByGroup(args, wide, dat):
    """
    Runs Count Values by group

    :Arguments:
        :type args: string
        :param args: command line args

        :type wide: pandas DataFrame
        :param wide: wide dataset

        :type dat: pandas DataFrame
        :param dat: wide and design datasets
    """
    #Open PDF pages
    with PdfPages(args.figure) as pdfOut:

        #Open CV dataframe
        CV=pd.DataFrame(index=wide.index)

        # Split design file by treatment group
        for title, group in dat.design.groupby(args.group):

            # Filter the wide file into a new dataframe
            currentFrame=wide[group.index]

            # Change dat.sampleIDs to match the design file
            dat.sampleIDs=group.index

            # Get CV and CVcutoff
            CV['cv_'+title], CVcutoff=setflag(args, currentFrame, dat, groupName=title)

        # Get max
        CV['cv']=CV.apply(np.max, axis=1)

        if not args.CVcutoff:
            CVcutoff=np.nanpercentile(CV['cv'].values, q=90)
            CVcutoff=round(CVcutoff, -int(floor(log(abs(CVcutoff), 10))) + 2)
        else:
            CVcutoff=float(args.CVcutoff)

        # Select palette
        ch=colorHandler('tableau','Tableau_10')
        logger.info(u"Using tableau Tableau_10 palette")

        # Get colors
        des,ucolor,combname=ch.getColors(dat.design,[args.group])
        logger.info("Plotting Data")

        # Split design file by treatment group
        for title, group in dat.design.groupby(args.group):

            # Open figure handler
            fh=figureHandler(proj='2d')

            # Get xmin
            xmin=-np.nanpercentile(CV['cv_'+title].values,99)*0.2

            # Get xmax
            xmax=np.nanpercentile(CV['cv_'+title].values,99)*1.5

            # Plot histogram
            hist.serHist(ax=fh.ax[0],dat=CV['cv_'+title],color='grey',normed=1,
                        range=(xmin,xmax),bins=15)

            # Plot density plot
            dist.plotDensityDF(data=CV['cv_'+title],ax=fh.ax[0],lb="CV density",
                        colors=ucolor[title])

            # Plot cutoff
            lines.drawCutoffVert(ax=fh.ax[0],x=CVcutoff,
                                lb="Cutoff at: {0}".format(CVcutoff))

            # Plot legend
            fh.makeLegendLabel(ax=fh.ax[0])

            # Give format to the axis
            fh.formatAxis(yTitle='Density',xlim=(xmin,xmax), ylim="ignore",
                figTitle = "Density Plot of Coefficients of Variation in {0} {1}".format(args.group,title))

            # Shrink figure to fit legend
            fh.shrink()

            # Add plot to PDF
            fh.addToPdf(pdfPages=pdfOut)

        # Open new figureHandler instance
        fh=figureHandler(proj='2d')

        #Get xmin
        xmin=-np.nanpercentile(CV['cv'].values,99)*0.2

        #Get xmax
        xmax=np.nanpercentile(CV['cv'].values,99)*1.5

        # Create flag file instance
        flag=Flags(index=CV['cv'].index)

        # Split design file by treatment group
        for title, group in dat.design.groupby(args.group):
            
            # Plot density plot
            dist.plotDensityDF(data=CV["cv_"+title],ax=fh.ax[0],colors=ucolor[title],
                                lb="CV density in group {0}".format(title))

            # Create new flag row for each group
            flag.addColumn(column="flag_feature_big_CV_{0}".format(title),
                         mask=((CV['cv_'+title].get_values() > CVcutoff) |
                          CV['cv_'+title].isnull()))

        # Plot cutoff
        lines.drawCutoffVert(ax=fh.ax[0],x=CVcutoff,lb="Cutoff at: {0}".format(CVcutoff))

        # Plot legend
        fh.makeLegendLabel(ax=fh.ax[0])

        # Give format to the axis
        fh.formatAxis(yTitle="Density", xlim=(xmin,xmax),
            figTitle="Density Plot of Coefficients of Variation by {0}".format(args.group))

        # Shrink figure
        fh.shrink()
        
        # Add figure to PDF
        fh.addToPdf(pdfPages=pdfOut)

    # Write flag file
    flag.df_flags.to_csv(args.flag, sep='\t')

def setflag(args, wide, dat, groupName=''):
    """
    Runs Count Values by group

    :Arguments:
        :type args: string
        :param args: command line args

        :type wide: pandas DataFrame
        :param wide: wide dataset

        :type dat: pandas DataFrame
        :param dat: wide and design datasets

        :type groupName: string
        :param groupName: name of optional group
    """
    # Round all values to 3 significant digits
    DATround=wide.applymap(lambda x: x)

    # Create empty dataset with metabolites names as index
    DATstat=pd.DataFrame(index=DATround.index)

    #Calculate STD
    DATstat["std"] =DATround.apply(np.std, axis=1)

    #Calculate for mean
    DATstat["mean"]=DATround.apply(np.mean, axis=1)

    DATstat["cv"]  =abs(DATstat["std"] / DATstat["mean"])

    if not args.CVcutoff:
        CVcutoff=np.nanpercentile(DATstat['cv'].values, q=90)

        CVcutoff=round(CVcutoff, -int(floor(log(abs(CVcutoff), 10))) + 2)
    else:
        CVcutoff=float(args.CVcutoff)

    # Plot CVs
    logger.info("Plotting Data")
    if groupName == "":
        # Initiate figure instance
        fh=figureHandler(proj="2d")

        # Calculate xmin
        xmin=-np.nanpercentile(DATstat['cv'].values,99)*0.2

        # Calculate xmax
        xmax=np.nanpercentile(DATstat['cv'].values,99)*1.5

        # Plot histogram
        hist.quickHist(ax=fh.ax[0],dat=DATstat['cv'],color="grey")

        # Plot density
        dist.plotDensityDF(data=DATstat['cv'],ax=fh.ax[0],colors='g',lb="CV density")

        # Ploot cutoff
        lines.drawCutoffVert(ax=fh.ax[0],x=CVcutoff,lb="Cutoff at: {0}".format(CVcutoff))

        # Plot legend
        fh.makeLegendLabel(ax=fh.ax[0])

        # Give format to axis
        fh.formatAxis(figTitle="Density Plot of Coefficients of Variation",
                    yTitle="Density",xlim=(xmin,xmax),ylim="ignore")

        # Shirnk plot
        fh.shrink()

        # Export figure
        fh.export(args.figure)

        # Create flag instance
        flag=Flags(index=DATstat.index)

        # Create new flag column with flags
        flag.addColumn(column='flag_feature_big_CV',
                    mask=((DATstat['cv'].get_values() > CVcutoff) | 
                    DATstat['cv'].isnull()))

        # Write output
        flag.df_flags.to_csv(args.flag, sep='\t')
    else:
        return DATstat['cv'], CVcutoff

def main(args):
    """ Function to input all the arguments"""
    # Import data
    dat=wideToDesign(args.input, args.design, args.uniqID, group=args.group)

    # Use group separation or not depending on user input
    if not args.group:
        logger.info("running CV without groups")
        setflag(args, dat.wide, dat)
    else:
        logger.info("running CV with groups")
        dat.removeSingle()
        setflagByGroup(args, dat.wide, dat)
   
    logger.info("DONE")

if __name__ == '__main__':
    # Command line options
    args=getOptions()

    # Logger
    logger=logging.getLogger()
    sl.setLogger(logger)

    # Main
    main(args)

