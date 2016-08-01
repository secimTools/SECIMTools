#!/usr/bin/env python
################################################################################
# SCRIPT: summarizeFlags.py
# 
# AUTHOR: Miguel Ibarra Arellano (miguelib@ufl.edu)
# 
# DESCRIPTION: This script takes a f;ag format file (wide), a flag file and a 
# unique ID for that file.  And its going to create a summary of the flags 
# calculating the sum, mean, any and all. Any wil tell you if any of the flags is
# on and all will tell you if all the flags are on.
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

#Standard Libraries
import logging
import argparse
from argparse import RawDescriptionHelpFormatter

#Add-on Libraries
import pandas as pd

#Local Libraries
from interface import wideToDesign
from flags import Flags
import logger as sl



"""Function to pull arguments"""
def getOptions():
    parser = argparse.ArgumentParser(description="Drops rows or columns given \
                                    an specific cut value and condition")
    
    required = parser.add_argument_group(title='Required Input', 
                                        description='Requiered input to the \
                                        program.')
    required.add_argument('-f',"--flags",dest="flagFile", action="store",
                        required=True,help="Flag file.")
    required.add_argument('-id',"--ID",dest="uniqID",action="store",
                        required=True,help="Name of the column with unique \
                        identifiers.")

    output = parser.add_argument_group(title='Output', description='Output of \
                                        the script.')
    output.add_argument('-os',"--outSummary",dest="outSummary",action="store",
                        required=True,help="Output file for Drops.")

    args = parser.parse_args()
    return(args);

def main():
    #Gettign arguments from parser
    args = getOptions()

    #Stablishing logger
    logger = logging.getLogger()
    sl.setLogger(logger)

    #Starting script
    logger.info(u'Importing data with following parameters: \
        \n\tFlags: {0}\
        \n\tID: {1}\
        \n\tSummary: {2}'.format(args.flagFile,args.uniqID,args.outSummary))

    # Convert flag file to DataFrame
    df_flags = pd.DataFrame.from_csv(args.flagFile, sep='\t')

    #Creating flag object
    df_offFlags = Flags(index=df_flags.index)

    #Get sum flags colum
    flagSum = df_flags.sum(axis=1)
    df_offFlags.addColumn("flag_sum",flagSum)

    #Get mean flags column
    df_offFlags.df_flags.loc[:,"flag_mean"] = df_flags.mean(axis=1)

    #Calculate flag at least one off
    maskAny = df_flags.any(axis=1)
    df_offFlags.addColumn("flag_any_off", maskAny)

    #Calculate flag all off
    maskAll = df_flags.all(axis=1)
    df_offFlags.addColumn("flag_all_off", maskAll)

    #Concatenate flags and summary flags
    df_offFlags = pd.concat([df_flags,df_offFlags.df_flags],axis=1)

    #Output flags
    df_offFlags.to_csv(args.outSummary, sep='\t')

    # Finishing script
    logger.info("Script complete.")

if __name__ == '__main__':
    main()
    