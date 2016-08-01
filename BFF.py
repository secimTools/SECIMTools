#!/usr/bin/env python
#Coded by ibarrarellano@gmail.com
#2016/Apr/19

#Standard Libraries
import argparse
import os
import math
import logging

#AddOn Libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Local Packages
from interface import wideToDesign
from flags import Flags
import logger as sl

def getOptions():
    """Function to pull arguments"""
    parser = argparse.ArgumentParser(description="Takes a peak area/heigh \
                                     dataset and calculates the LOD on it ")

    #Standar input for SECIMtools
    standar = parser.add_argument_group(title='Standard input', description= 
                                        'Standard input for SECIM tools.')
    standar.add_argument("-i","--input",dest="input", action='store', 
                        required=True, help="Input dataset in wide format.")
    standar.add_argument("-d","--design",dest="design", action='store', 
                        required=True, help="Design file.")
    standar.add_argument("-id","--uniqID",dest="uniqID",action="store",
                        required=True, help="Name of the column with unique \
                        dentifiers.")
    standar.add_argument("-g","--group", dest="group", action='store', 
                        required=True, help="Name of column in design file \
                        with Group/treatment information.")

    #Output Paths
    output = parser.add_argument_group(title='Output paths', description=
                                       "Paths for the output files")
    output.add_argument("-f","--outflags",dest="outflags",action="store",
                        required=True,help="Output path for flags file[CSV]")
    output.add_argument("-b","--outbff",dest="outbff",action="store",
                        required=True,help="Output path for bff file[CSV]")

    #Optional Input
    optional = parser.add_argument_group(title='Optional input', 
                                         description="Changes parameters for \
                                         the prorgram")
    optional.add_argument("-bv","--bff" ,dest="bff", action='store', 
                         required=False, default=5000, 
                         help="Default BFF value [default 5000]")
    optional.add_argument("-bn","--blank",dest="blank", action="store",
                        required=False, default="blank",
                        help="name of the column with the blanks")
    optional.add_argument("-c","--criteria",dest="criteria",action="store",
                        required=False, default=100,type=int,
                        help="Value of the criteria to selct")
    optional.add_argument("-log", "--log", dest="log", action='store', 
                         required=False, default=False, 
                         help="Path and name of log file")

    args = parser.parse_args()
    return (args)

def calculateLOD(row,defaultLOD):
    lod = round(np.average(row)+(3*np.std(row,ddof=1)),3)
    if lod > 0:
        return lod
    else:
        lod=5000
        return lod

def gettingBFF(row):
    groups = row[row.index!="lod"]
    bff = groups.apply(lambda group: ((group-row["lod"])/row["lod"]))
    return bff

def main():
    #Import data
    global args
    args = getOptions()

    #Setting logger
    logger = logging.getLogger()
    sl.setLogger(logger)
    logger.info(u"""Importing data with following parameters:
                Input: {0}
                Design: {1}
                uniqID: {2}
                group: {3}
                blank: {4}
                """.format(args.input, args.design, args.uniqID, args.group,
                    args.blank))

    #Importing data
    dat = wideToDesign(args.input, args.design, args.uniqID, args.group)
    dat.wide = dat.wide.applymap(float)

    groups=dat.levels
    #Calculating the means per group
    nob_means = pd.DataFrame(index=dat.wide.index)#,columns=dat.levels)
    for group in dat.levels:
        if group==args.blank:
            blanks_df=dat.wide[dat.design.index[dat.design[dat.group]==group]]
        else:
            currentgroup_df=dat.wide[dat.design.index[dat.design[dat.group]==group]]
            nob_means[group]=currentgroup_df.mean(axis=1)

    #Calculating the LOD
    logger.info(u"""Calculating limit of detection for each group, if limit
                of detection equals 0 default value [{0}] will be used.
                """.format(args.bff))
    nob_means.loc[:,"lod"]=blanks_df.apply(lambda row:calculateLOD(row,args.bff),
                                            axis=1)

    #Aplying BFF
    nob_bff=pd.DataFrame(index=dat.wide.index,columns=nob_means.columns)
    logger.info(u"""Comparing value of limit of detection to criteria [{0}]."""
                .format(args.criteria))
    nob_bff=nob_means.apply(gettingBFF,axis=1)

    #Creating objet with flags
    df_offFlags = Flags(index=nob_bff.index)

    logger.info(u"""Creating flags.""")
    for group in nob_bff:
        #Get mask for values
        mask = (nob_bff[group] > args.criteria)
        
        #Add Falg column
        df_offFlags.addColumn(column='flag_bff_'+group+'_off', mask=mask==False)

    #Calculate flag at least one off
    maskAny = df_offFlags.df_flags.any(axis=1)
    df_offFlags.addColumn('flag_bff_any_off', maskAny)

    #Calculate flag all off
    maskAll = df_offFlags.df_flags.all(axis=1)
    df_offFlags.addColumn('flag_bff_all_off', maskAll)

    #Outputting files
    logger.info(u"""Output limit of detection values file.""")
    nob_bff.to_csv(os.path.abspath(args.outbff), sep='\t')
    logger.info(u"""Output flag file.""")
    df_offFlags.df_flags.to_csv(os.path.abspath(args.outflags), sep='\t')

if __name__ == '__main__':
    main()