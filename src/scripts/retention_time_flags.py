#!/usr/bin/env python
################################################################################
# DATE: 2017/06/23
#
# SCRIPT: retention_time_flags.py
#
# VERSION: 2.3
# 
# AUTHOR: Miguel A. Ibarra (miguelib@ufl.edu), 
# Edited by Alexander Kirpich (akirpich@ufl.edu)
# 
# DESCRIPTION: Retention time (RT) is related to the location of the compound.
#    Thus, across samples, the retention times for one compound do not vary much 
#    if it's consistent. Therefore, we use various measurements for the variation
#    of retention times to detect whether the compound is consistent
#
################################################################################
# Import built-in libraries
import os
import logging
import argparse
import warnings
from math import log, floor
from argparse import RawDescriptionHelpFormatter

# Import add-on libraries
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Import local data libraries
from secimtools.dataManager import logger as sl
from secimtools.dataManager.flags import Flags
from secimtools.dataManager.interface import wideToDesign

# Import local plotting libraries
from secimtools.visualManager import module_hist as hist
from secimtools.visualManager import module_lines as lines
from secimtools.visualManager import module_distribution as dist
from secimtools.visualManager.manager_color import colorHandler
from secimtools.visualManager.manager_figure import figureHandler


def getOptions(myopts=None):
    """Function to pull in arguments"""

    description="""Retention time (RT) is related to the location of the compound.
    Thus, across samples, the retention times for one compound do not vary much 
    if it's consistent. Therefore, we use various measurements for the variation
    of retention times to detect whether the compound is consistent.

    Retention time flags include,
        flag_RT_Q90Q10_outlier:     Flagged when the difference between the 90th
                                    and the 10th percentile of RT of the compound
                                    is greater than 0.2.
        flag_RT_Q95Q05_outlier:     Flagged when the difference between the 95th
                                    and the 5th percentile of RT of the compound
                                    is greater than 0.2.
        flag_RT_max_gt_threshold:   Flagged when the difference between the maximum
                                    and the median of the RT of the compound is 
                                    greater than 0.1
        flag_RT_min_lt_threshold:   Flagged when the difference between the minimum
                                    and the median of the RT of the compound is 
                                    greater than 0.1
        flag_RT_min_max_outlier:    Flagged when the minimum or the maximum is more
                                    than 3 times the standard deviation away from
                                    the mean of the RT of the compound
        flag_RT_big_CV:             A flag for large CV of the RT

    """

    parser=argparse.ArgumentParser(description=description, 
                                formatter_class=RawDescriptionHelpFormatter)
    # Standard Input
    standard=parser.add_argument_group(title='Standard input', 
                        description='Standard input for SECIM tools.')
    standard.add_argument('-i',"--input", dest="input", action='store', 
                        required=True, help="Input dataset in wide format.")
    standard.add_argument('-d',"--design", dest="design", action='store', 
                        required=True, help="Design file.")
    standard.add_argument('-id',"--ID",dest="uniqID",action='store',required=True,
                        help="Name of the column with unique identifiers.")
    # Tool Input
    tool=parser.add_argument_group(title='Tool Input',
                                description='Input specific for the tool.')
    tool.add_argument('-m',"--minutes",dest="minutes",action='store', 
                        required=False,default=0.2,type=float, 
                        help="Value cutoff. The default is .2 minutes assuming data are in minutes.")
    tool.add_argument("--pctl", dest="p90p10", action='store_true', 
                        required=False, default=False, 
                        help="""The difference is calculated by 95th percentile 
                        and 5th percentile by default. If you add this argument,
                         it uses 90th percentile and 10th percentile. [optional]""")
    tool.add_argument("-cv","--CVcutoff", dest="CVcutoff", action='store', 
                        required=False, default=False, type=float, 
                        help="""The default CV cutoff will flag 10 percent of 
                        the rowIDs with larger CVs. If you want to set a CV 
                        cutoff, put the number here. [optional]""")
    # Tool Output
    output=parser.add_argument_group(title='Output',
                                description='Output parameters.')
    output.add_argument('-f',"--figure",dest="figure",action='store', 
                        required=True,default='CVplot', 
                        help="Name of the output TSV for CV plots.")
    output.add_argument('-fl',"--flag",dest="flag",action='store',
                        required=True,default='flag', 
                        help="Name of the output PDF for RT flags.")
    # Plot Options
    plot = parser.add_argument_group(title='Plot options')
    plot.add_argument("-pal","--palette",dest="palette",action='store',required=False, 
                        default="qualitative", help="Name of the palette to use.")
    plot.add_argument("-col","--color",dest="color",action="store",required=False, 
                        default="Set1_9", help="Name of a valid color"
                        " scheme on the selected palette")

    if myopts:
        args=parser.parse_args(myopts)
    else:
        args=parser.parse_args()

    # Standardize paths
    args.flag   = os.path.abspath(args.flag)
    args.input  = os.path.abspath(args.input)
    args.design = os.path.abspath(args.design)
    args.figure = os.path.abspath(args.figure)

    return(args)

def ifZero(x):
    if x == 0:
        value=np.nan
    else:
        value=x
    return value

def runStats(args, wide, dat):
    logger.info("Running RT Flag")
    # Round retention time to 2 decimals
    RTround=wide.applymap(lambda x: ifZero(x))
    RTround=RTround.applymap(lambda x: round(x, 2))

    # Get percentiles, min, max, mean, median
    RTstat=pd.DataFrame(index=RTround.index)
    RTstat['min']   =RTround.apply(np.min, axis=1)
    RTstat['max']   =RTround.apply(np.max, axis=1)
    RTstat['p95']   =RTround.apply(np.nanpercentile, q=95, axis=1)
    RTstat['p90']   =RTround.apply(np.nanpercentile, q=90, axis=1)
    RTstat['p10']   =RTround.apply(np.nanpercentile, q=10, axis=1)
    RTstat['p05']   =RTround.apply(np.nanpercentile, q= 5, axis=1)
    RTstat['std']   =RTround.apply(np.std, axis=1, ddof=1)
    RTstat['mean']  =RTround.apply(np.mean, axis=1)
    RTstat['median']=RTround.apply(np.median, axis=1)
    RTstat['cv']    =RTstat['std'] / RTstat['mean']
    RTstat['p95p05']=RTstat['p95'] - RTstat['p05']
    RTstat['p90p10']=RTstat['p90'] - RTstat['p10']

    # Return RTstat
    return (RTstat)

def plotCV(data,cutoff,pdf):
    # Creating figure instance
    fh=figureHandler(proj='2d')

    # Getting xmin and xmax
    xmin=-np.nanpercentile(data['cv'].values,99)*0.2
    xmax=np.nanpercentile(data['cv'].values,99)*1.5

    # plotting histogra,
    hist.serHist(ax=fh.ax[0],dat=data['cv'],range=(xmin, xmax), bins=15, 
                normed=1, color=palette.mpl_colors[8])

    # Plotting distribution
    dist.plotDensityDF(data=data['cv'],ax=fh.ax[0],colors=palette.mpl_colors[1],
                        lb='CV density')

    # Draw cutoffs
    lines.drawCutoffVert(ax=fh.ax[0],x=cutoff,cl=palette.mpl_colors[0],
                        lb="Cutoff at:\n{0}".format(cutoff))

    # making legend
    fh.makeLegendLabel(ax=fh.ax[0])

    # Formating axis
    fh.formatAxis(yTitle='Density',xlim=(xmin,xmax), ylim="ignore",
            figTitle="Density Plot of Coef. of Variation of the Retention Time")
            
    #Shrinking figure
    fh.shrink()

    #Export figure
    fh.addToPdf(dpi=600,pdfPages=pdf)

def main(args):
    """Main script """
    # Import data
    dat = wideToDesign(args.input, args.design, args.uniqID, logger=logger)

    # Run Stats
    logger.info("Calculatting stats")
    RTstat = runStats(args, dat.wide, dat)

    # Cleaning from missing data
    dat.dropMissing()

    # Set RT Flags
    logger.info("Creatting Flags")
    flag=Flags(index=RTstat.index)
    if args.p90p10:
        flag.addColumn(column='flag_RT_Q90Q10_outlier',
                        mask =(RTstat['p90p10'] > args.minutes))
    else:
        flag.addColumn(column='flag_RT_Q95Q05_outlier',
                        mask =(RTstat['p95p05'] > args.minutes))

    flag.addColumn(column='flag_RT_max_gt_threshold',
                    mask =(RTstat['max']-RTstat['median']>args.minutes/2))

    flag.addColumn(column='flag_RT_min_lt_threshold',
                    mask =(RTstat['min']-RTstat['median']<-args.minutes/2))

    flag.addColumn(column='flag_RT_min_max_outlier',
                    mask =((RTstat['max']-RTstat['mean']>3*RTstat['std']) |
                              (RTstat['min']-RTstat['mean']<-3*RTstat['std'])))

    if not args.CVcutoff:
        CVcutoff = np.nanpercentile(RTstat['cv'].values, q=90)
    else:
        CVcutoff = np.nanpercentile(RTstat['cv'].values, q=(1-args.CVcutoff)*100 )

    CVcutoff = round(CVcutoff, -int(floor(log(CVcutoff, 10))) + 2)


    flag.addColumn(column='flag_RT_big_CV', 
                    mask=(RTstat['cv'] > CVcutoff))


    # Plot data
    logger.info("Plotting data")
    with PdfPages(args.figure) as pdfOut:
        plotCV(data=RTstat,cutoff=CVcutoff,pdf=pdfOut)

    # Output flags
    flag.df_flags.to_csv(args.flag, sep="\t")
    logger.info("Script Complete!")


if __name__ == '__main__':

    # Command line options
    args=getOptions()

    #Set logger
    logger=logging.getLogger()
    sl.setLogger(logger)

    # Stablishing color palette
    palette = colorHandler(pal=args.palette, col=args.color)
    logger.info(u"Using {0} color scheme from {1} palette".format(args.color,
                args.palette))

    # Setting np warrnings off!!!!!
    np.seterr(invalid='ignore')

    # Ignore the rest of the warnings!!!!
    warnings.filterwarnings("ignore")#, category=DeprecationWarning)

    #Main
    main(args)

