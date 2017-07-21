#!/usr/bin/env python
################################################################################
# DATE: 2017/03/07
# 
# MODULE: blank_feature_filtering_flags.py
#
# VERSION: 1.1
# 
# AUTHOR: Miguel Ibarra (miguelib@ufl.edu)
#
# DESCRIPTION: This tool Blank Feature Filtering on wide data.
#
################################################################################
# Import built-in libraries
import os
import logging
import argparse

# Import add-on libraries
import numpy as np
import pandas as pd

# Import local data libraries
from secimtools.dataManager import logger as sl
from secimtools.dataManager.flags import Flags
from secimtools.dataManager.interface import wideToDesign

def getOptions():
    """Function to pull arguments"""
    parser = argparse.ArgumentParser(description="Take a peak area or height dataset " \
                                                 "and calculate the limits of detection for features based on the blank samples. ")
    # Standard Input
    standar = parser.add_argument_group(title='Standard input', description= 
                                        'Standard input for SECIM tools.')
    standar.add_argument("-i","--input",dest="input", action='store', 
                        required=True, help="Input dataset in wide format.")
    standar.add_argument("-d","--design",dest="design", action='store', 
                        required=True, help="Design file.")
    standar.add_argument("-id","--uniqID",dest="uniqID",action="store",
                        required=True, help="Name of the column with unique" \
                        "dentifiers.")
    standar.add_argument("-g","--group", dest="group", action='store', 
                        required=True, help="Name of column in design file" \
                        "with Group/treatment information.")
    # Tool Input
    tool = parser.add_argument_group(title='Optional input', 
                                         description="Changes parameters for "\
                                         "the prorgram")
    tool.add_argument("-bv","--bff" ,dest="bff", action='store', 
                         required=False, default=5000, type=int,
                         help="Default BFF value [default 5000]")
    tool.add_argument("-bn","--blank",dest="blank", action="store",
                        required=False, default="blank",
                        help="name of the column with the blanks")
    tool.add_argument("-c","--criteria",dest="criteria",action="store",
                        required=False, default=100,type=int,
                        help="Value of the criteria to selct")
    # Tool Output
    output = parser.add_argument_group(title='Output paths', description=
                                       "Paths for the output files")
    output.add_argument("-f","--outflags",dest="outflags",action="store",
                        required=True,help="Output path for flags file[CSV]")
    output.add_argument("-b","--outbff",dest="outbff",action="store",
                        required=True,help="Output path for bff file[CSV]")
    args = parser.parse_args()

    # Standardize paths
    args.input    = os.path.abspath(args.input)
    args.design   = os.path.abspath(args.design)
    args.outflags = os.path.abspath(args.outflags)
    args.outbff   = os.path.abspath(args.outbff)

    # Returning arguments
    return (args)

def main(args):
    #Importing data
    logger.info("Importing data with the Interface")
    dat = wideToDesign(args.input, args.design, args.uniqID, args.group, 
                        logger=logger)
    
    # Cleaning from missing data
    dat.dropMissing()

    # Calculate the means of each group but blanks
    logger.info("Calcualting group means")
    df_nobMeans = pd.DataFrame(index=dat.wide.index)
    for name, group in dat.design.groupby(dat.group):
        if name == args.blank:
            df_blank = dat.wide[group.index].copy()
        else:
            df_nobMeans[name] = dat.wide[group.index].mean(axis=1)

    # Calculating the LOD
    # Calculates the average of the blanks plus3 times the SD of the same.
    # If value calculated is 0 then use the default lod (default = 5000)
    # NOTE: ["lod"]!=0 expression represents that eveything that is not 0 is fine
    # and shoud remain as it is, and eveything that is 0  shoud be replaced
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.where.html
    logger.info("Calculating limit of detection for each group default value [{0}].".format(args.bff))
    df_blank.loc[:,"lod"] = np.average(df_blank,axis=1)+(3*np.std(df_blank,ddof=1,axis=1))
    df_blank["lod"].where(df_blank["lod"]!=0, args.bff, inplace=True)

    # Apoply the limit of detection to the rest of the data, these values will be
    # compared agains the criteria value for flagging.
    logger.info("Comparing value of limit of detection to criteria [{0}].".format(args.criteria))
    nob_bff = pd.DataFrame(index=dat.wide.index, columns=df_nobMeans.columns)
    for group in nob_bff:
        nob_bff.loc[:,group]=(df_nobMeans[group]-df_blank["lod"])/df_blank["lod"]

    # We create flags based on the criteria value (user customizable)
    logger.info("Creating flags.")
    df_offFlags = Flags(index=nob_bff.index)
    for group in nob_bff:
        df_offFlags.addColumn(column = 'flag_bff_'+group+'_off', 
                              mask   = (nob_bff[group] < args.criteria))

    # Output BFF values and flags
    nob_bff.to_csv(args.outbff, sep='\t')
    df_offFlags.df_flags.to_csv(args.outflags, sep='\t')
    logger.info("Script Complete!")

if __name__ == '__main__':
    #Import data
    args = getOptions()

    #Setting logger
    logger = logging.getLogger()
    sl.setLogger(logger)
    logger.info("Importing data with following parameters:"\
                "\n\tInput: {0}"\
                "\n\tDesign: {1}"\
                "\n\tuniqID: {2}"\
                "\n\tgroup: {3}"\
                "\n\tblank: {4}".format(args.input, args.design, args.uniqID, args.group,
                    args.blank))

    # Calling main script
    main(args)
