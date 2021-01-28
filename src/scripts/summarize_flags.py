#!/usr/bin/env python
################################################################################
# DATE: 2017/03/09
#
# SCRIPT: summarize_flags.py
# 
# AUTHOR: Miguel A. Ibarra-Arellano (miguelib@ufl.edu)
# 
# DESCRIPTION: This script takes a f;ag format file (wide), a flag file and a 
# unique ID for that file.  And its going to create a summary of the flags 
# calculating the sum, mean, any and all. Any wil tell you if any of the flags 
# is on and all will tell you if all the flags are on.
#
# INPUT: 
#       Flag file:
#               Wide formatted file with flags
#
# OUTPUT:
#       Flag file:
#               Wide formatted file with the original flags and the summary of\
#               them. (Per Row)
#
################################################################################
# Import built-in libraries
import os
import logging
import argparse
from argparse import RawDescriptionHelpFormatter

# Import add-on libraries
import pandas as pd

# Import local data libraries
from secimtools.dataManager import logger as sl
from secimtools.dataManager.flags import Flags
from secimtools.dataManager.interface import wideToDesign

def getOptions():
    """Function to pull arguments"""
    parser = argparse.ArgumentParser(description="Drops rows or columns given "\
                                    "an specific cut value and condition")
    
    required = parser.add_argument_group(title="Required Input", 
                                        description="Requiered input to the "\
                                        "program.")
    # Required input
    required.add_argument('-f',"--flags",dest="flagFile", action="store",
                        required=True,help="Flag file.")
    required.add_argument('-id',"--ID",dest="uniqID",action="store",
                        required=True,help="Name of the column with unique "\
                        "identifiers.")
    # Tool output
    output = parser.add_argument_group(title="Output", description="Output of "\
                                        "the script.")
    output.add_argument('-os',"--outSummary",dest="outSummary",action="store",
                        required=True,help="Output file for Summary.")
    args = parser.parse_args()

    # Standardize paths
    args.flagFile   = os.path.abspath(args.flagFile)
    args.outSummary = os.path.abspath(args.outSummary)

    return(args);

def main(args):
    # Convert flag file to DataFrame
    df_inp_flags = pd.read_csv(args.flagFile, sep='\t')

    #Creating flag object
    offFlags_df = Flags(index=df_inp_flags.index)

    #Get flags for sum,mean, any and all
    logger.info("Creating flags")
    offFlags_df.addColumn("flag_sum",df_inp_flags.sum(axis=1))
    offFlags_df.df_flags.loc[:,"flag_mean"]=df_inp_flags.mean(axis=1)
    offFlags_df.addColumn("flag_any_off", df_inp_flags.any(axis=1))
    offFlags_df.addColumn("flag_all_off", df_inp_flags.all(axis=1))

    #Concatenate flags and summary flags
    offFlags_df = pd.concat([df_inp_flags,offFlags_df.df_flags],axis=1)

    #Output flags
    offFlags_df.to_csv(args.outSummary, sep='\t')

    # Finishing script
    logger.info("Script complete.")

if __name__ == '__main__':
    # Getting arguments from parser
    args = getOptions()

    # Stablishing logger
    logger = logging.getLogger()
    sl.setLogger(logger)

    # Starting script
    logger.info(u"Importing data with following parameters: "\
        "\n\tFlags: {0}"\
        "\n\tID: {1}"\
        "\n\tSummary: {2}".format(args.flagFile,args.uniqID,args.outSummary))

    # Main script
    main(args)
