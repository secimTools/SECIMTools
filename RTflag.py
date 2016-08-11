#!/usr/bin/env python

# Built-in packages
import logging
import argparse
import os
import shutil
import tempfile
from argparse import RawDescriptionHelpFormatter
from math import log, floor

# Add-on packages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local packages
from flags import Flags
from interface import wideToDesign
import logger as sl

from manager_color import colorHandler
from manager_figure import figureHandler
import module_lines as lines
import module_hist as hist
import module_distribution as dist


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

    group1=parser.add_argument_group(title='Standard input', 
                                description='Standard input for SECIM tools.')
    group1.add_argument('-i',"--input", dest="input", action='store', 
                        required=True, help="Input dataset in wide format.")
    group1.add_argument('-d',"--design", dest="design", action='store', 
                        required=True, help="Design file.")
    group1.add_argument('-id',"--ID",dest="uniqID",action='store',required=True,
                        help="Name of the column with unique identifiers.")
    group1.add_argument('-f',"--figure",dest="figure",action='store', 
                        required=True,default='CVplot', 
                        help="Name of the output TSV for CV plots.")
    group1.add_argument('-fl',"--flag",dest="flag",action='store',
                        required=True,default='flag', 
                        help="Name of the output PDF for RT flags.")
    group1.add_argument('-m',"--minutes",dest="minutes",action='store', 
                        required=False,default=0.2,type=float, 
                        help="Percentile cutoff. The default is .2 minutes")

    group2=parser.add_argument_group(title='Optional input',
                                description='Optional input for SECIM tools.')
    group2.add_argument("--pctl", dest="p90p10", action='store_true', 
                        required=False, default=False, 
                        help="""The difference is calculated by 95th percentile 
                        and 5th percentile by default. If you add this argument,
                         it uses 90th percentile and 10th percentile. [optional]""")
    group2.add_argument("--CVcutoff", dest="CVcutoff", action='store', 
                        required=False, default=False, type=float, 
                        help="""The default CV cutoff will flag 10 percent of 
                        the rowIDs with larger CVs. If you want to set a CV 
                        cutoff, put the number here. [optional]""")

    if myopts:
        args=parser.parse_args(myopts)
    else:
        args=parser.parse_args()

    return(args)

def ifZero(x):

    if x == 0:
        value=np.nan
    else:
        value=x
    return value

def setflag(args, wide, dat, dir):
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
    RTstat['std']   =RTround.apply(np.std, axis=1)
    RTstat['mean']  =RTround.apply(np.mean, axis=1)
    RTstat['median']=RTround.apply(np.median, axis=1)
    RTstat['cv']    =RTstat['std'] / RTstat['mean']
    RTstat['p95p05']=RTstat['p95'] - RTstat['p05']
    RTstat['p90p10']=RTstat['p90'] - RTstat['p10']

    # Set RT flags
    flag=Flags(index=RTround.index)
    if args.p90p10:
        flag.addColumn(column='flag_RT_Q90Q10_outlier',
                        mask  =(RTstat['p90p10'] > args.minutes))
    else:
        flag.addColumn(column='flag_RT_Q95Q05_outlier',
                        mask  =(RTstat['p95p05'] > args.minutes))

    flag.addColumn(column='flag_RT_max_gt_threshold',
                    mask  =(RTstat['max']-RTstat['median']>args.minutes/2))

    flag.addColumn(column='flag_RT_min_lt_threshold',
                    mask  =(RTstat['min']-RTstat['median']<-args.minutes/2))

    flag.addColumn(column='flag_RT_min_max_outlier',
                    mask =((RTstat['max']-RTstat['mean']>3*RTstat['std']) |
                              (RTstat['min']-RTstat['mean']<-3*RTstat['std'])))

    if not args.CVcutoff:
        CVcutoff=np.nanpercentile(RTstat['cv'].values, q=90)
        CVcutoff=round(CVcutoff, -int(floor(log(CVcutoff, 10))) + 2)
    else:
        CVcutoff=args.CVcutoff
    flag.addColumn(column='flag_RT_big_CV',
                    mask  =(RTstat['cv'] > CVcutoff))

    # Output flags
    flag.df_flags.to_csv(args.flag, sep="\t")

    # Plot RT CVs
    logger.info("Plotting Data")

    # Creating figure instance
    fh=figureHandler(proj='2d')

    # Getting xmin and xmax
    xmin=-np.nanpercentile(RTstat['cv'].values,99)*0.2
    xmax=np.nanpercentile(RTstat['cv'].values,99)*1.5

    # plotting histogra,
    hist.serHist(ax=fh.ax[0],dat=RTstat['cv'],range=(xmin, xmax), bins=15, 
                normed=1, color='grey')

    # Plotting distribution
    dist.plotDensityDF(data=RTstat['cv'],ax=fh.ax[0],colors='c',lb='CV density')

    # Draw cutoffs
    lines.drawCutoffVert(ax=fh.ax[0],x=CVcutoff,lb="Cutoff at:\n{0}".format(CVcutoff))

    # making legend
    fh.makeLegendLabel(ax=fh.ax[0])

    # Formating axis
    fh.formatAxis(yTitle='Density',xlim=(xmin,xmax), ylim="ignore",
            figTitle="Density Plot of Coef. of Variation of the Retention Time")
            
    #Shrinking figure
    fh.shrink()

    #Export figure
    fh.export(args.figure)
    logger.info("DONE")

def main(args):
    """ """
    directory=args.flag

    # Import data
    dat=wideToDesign(args.input, args.design, args.uniqID)

    # Only interested in samples
    wide=dat.wide[dat.sampleIDs]

    # Set RT flags
    setflag(args, wide, dat, dir=directory)


if __name__ == '__main__':

    # Command line options
    args=getOptions()

    #Set logger
    logger=logging.getLogger()
    sl.setLogger(logger)

    # Setting np warrnings off!!!!!
    np.seterr(invalid='ignore')

    #Main
    main(args)

