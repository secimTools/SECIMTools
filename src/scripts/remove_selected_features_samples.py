#!/usr/bin/env python
################################################################################
# SCRIPT: remove_selected_features_samples.py
# 
# LAST VERSION: Includes support to select the type of drop to perform instead
#               of automatically decide. 
#
# AUTHOR: Miguel Ibarra Arellano (miguelib@ufl.edu).
# 
# DESCRIPTION: This script takes a Wide format file (wide), a flag file and a 
# design file (only for drop by column) and drops either rows or columns for a 
# given criteria. This criteria could be either numeric,string or a flag (1,0). 
#
# OUTPUT:
#       Drop by row:
#                   Wide file with just the dropped rows
#                   Wide file without the dropped rows
#       Drop by column:
#                   Wide file with just the dropped columns
#                   Wide file without the dropped columns
#
################################################################################
# Import built-in libraries
import os
import re
import copy
import logging
import argparse
from argparse import RawDescriptionHelpFormatter

# Import add-on libraries
import pandas as pd

# Import local data libraries
from secimtools.dataManager import logger as sl
from secimtools.dataManager.interface import wideToDesign


"""Function to pull arguments"""
def getOptions():
    parser = argparse.ArgumentParser(description="Removes rows or columns from the data" \
                                    "using user-defined cut-offs.")
    # Standard Input
    standard = parser.add_argument_group(title='Required Input', 
                        description="Standard inputs for SECIM tools.")
    standard.add_argument('-i',"--input",dest="input", action="store",
                        required=True,help="Input dataset in wide format.")
    standard.add_argument('-d',"--design",dest="design", action="store",
                        required=True, help="Design file.")
    standard.add_argument('-id',"--ID",dest="uniqID",action="store",
                        required=True,help="Name of the column with unique "\
                        "identifiers.")
    # Tool Input
    tool = parser.add_argument_group(title="Tool Input", 
                        description="Tool especific input.")
    tool.add_argument('-f',"--flags",dest="flags", action="store",
                        required=True,help="Flag file.")
    tool.add_argument('-fft',"--flagfiletype",dest="flagfiletype",action='store',
                        required=True, default="row", choices=["row","column"],
                        help="Type of flag file")   
    tool.add_argument('-fid',"--flagUniqID",dest="flagUniqID",action="store",
                        required=False, default=False,help="Name of the column "\
                        "with unique identifiers in the flag files.")
    tool.add_argument('-fd',"--flagDrop",dest="flagDrop",action='store',
                        required=True, help="Name of the flag/field you want to"\
                        " access.")                                
    tool.add_argument('-val',"--value",dest="value",action='store',
                        required=False, default="1",help="Cut Value")
    tool.add_argument('-con',"--condition",dest="condition",action='store',
                        required=False, default="0",help="Condition for the cut" \
                        "where 0=Equal to, 1=Greater than and 2=less than.")
    # Tool Output
    output = parser.add_argument_group(title='Output', 
                        description="Output of the script.")
    output.add_argument('-ow',"--outWide",dest="outWide",action="store",required=True,
                        help="Output file without the Drops.")
    output.add_argument('-of',"--outFlags",dest="outFlags",action="store",
                        required=True,help="Output file for Drops.")
    args = parser.parse_args()

    # Standardize paths
    args.input    = os.path.abspath(args.input)
    args.flags    = os.path.abspath(args.flags)
    args.design   = os.path.abspath(args.design)
    args.outWide  = os.path.abspath(args.outWide)
    args.outFlags = os.path.abspath(args.outFlags)

    return(args);

def dropRows(df_wide, df_flags,cut_value, condition, args):
    """ 
    Drop rows in a wide file based on its flag file and the specified flag 
    values to keep.

    :Arguments:
        :type df_wide: pandas.DataFrame
        :param df: A data frame in wide format

        :type df_flags: pandas.DataFrame
        :param df: A data frame of flag values corresponding to the wide file

        :type cut_value: string
        :param args: Cut Value for evaluation

        :type condition: string
        :param args: Condition to evaluate

        :type args: argparse.ArgumentParser
        :param args: Command line arguments.

    :Returns:
        :rtype: pandas.DataFrame
        :returns: Updates the wide DataFrame with dropped rows and writes to a
            TSV.
        :rtype: pandas.DataFrame
        :returns: Fron wide DataFrame Dropped rows and writes to a TSV.
    """
    #Dropping flags from flag files, first asks for the type of value, then asks
    # for the diferent type of conditions new conditios can be added here
    if not((re.search(r'[a-zA-Z]+', cut_value))):
        cut_value = float(cut_value)
        if condition == '>':
            df_filtered =  df_flags[df_flags[args.flagDrop]<cut_value]
        elif condition == '<':
            df_filtered =  df_flags[df_flags[args.flagDrop]>cut_value]
        elif condition == '==':
            df_filtered =  df_flags[df_flags[args.flagDrop]!=cut_value]
        else:
            logger.error(u'The {0} is not supported by the program, please use <,== or >'.format(condition))
            quit()
    else:
        cut_value = str(cut_value)
        if condition == '==':
            df_filtered =  df_flags[df_flags[args.flagDrop]!=cut_value]
        else:
            logger.error(u'The {0} conditional is not supported for string flags, please use =='.format(condition))
            quit()

    #Create a mask over the original data to determinate what to delete
    mask = df_wide.index.isin(df_filtered.index)

    #Create a mask over the original flags to determinate what to delete
    mask_flags = df_flags.index.isin(df_filtered.index)

    # Use mask to drop values form original data
    df_wide_keeped = df_wide[mask]

    # Use mas to drop values out of original flags
    df_flags_keeped = df_flags[mask_flags]

    # Returning datasets
    return df_wide_keeped,df_flags_keeped

def dropColumns(df_wide, df_flags,cut_value, condition, args):
    """ 
    Drop columns in a wide file based on its flag file and the specified flag 
    values to keep.

    :Arguments:
        :type df_wide: pandas.DataFrame
        :param df: A data frame in wide format

        :type df_flags: pandas.DataFrame
        :param df: A data frame of flag values corresponding to the wide file

        :type cut_value: string
        :param args: Cut Value for evaluation

        :type condition: string
        :param args: Condition to evaluate

        :type args: argparse.ArgumentParser
        :param args: Command line arguments.

    :Returns:
        :rtype: pandas.DataFrame
        :returns: Updates the wide DataFrame with dropped columns and writes to 
                    a TSV.
        :rtype: pandas.DataFrame
        :returns: Fron wide DataFrame Dropped columns and writes to a TSV.
    """
    #Getting list of filtered columns from flag files
    if not(re.search(r'[a-zA-Z]+', cut_value)):
        cut_value = float(cut_value)
        if condition == '>':
            to_keep = df_flags.index[df_flags[args.flagDrop]<cut_value]
        elif condition == '<':
            to_keep = df_flags.index[df_flags[args.flagDrop]>cut_value]
        elif condition == '==':
            to_keep = df_flags.index[df_flags[args.flagDrop]!=cut_value]
        else:
            logger.error(u'The {0} is not supported by the program, please use <,== or >'.format(condition))
            quit()
    else:
        cut_value = str(cut_value)
        if condition == '==':
            to_keep = df_flags.index[df_flags[args.flagDrop]!=cut_value]
        else:
            logger.error(u'The {0} conditional is not supported for string flags, please use =='.format(condition))
            quit()

    # Subsetting
    dropped_flags = df_flags.T[to_keep].T
    df_wide = df_wide[to_keep]

    # Returning
    return df_wide,dropped_flags

def main(args):
    # Import data with the interface
    dat = wideToDesign(wide=args.input, design=args.design, 
                    uniqID=args.uniqID, logger=logger)

    # Cleaning from missing data
    dat.dropMissing()
    
    # Read flag file
    df_flags = pd.read_table(args.flags)

    # Select index on flag file if none then rise an error
    if args.flagUniqID:
        df_flags.set_index(args.flagUniqID, inplace=True)
    else:
        logger.error("Not flagUniqID provided")
        raise

    # Drop either rows or columns
    logger.info("Running drop flags by {0}".format(args.flagfiletype))
    if args.flagfiletype=="column":
        kpd_wide,kpd_flag = dropColumns(df_wide=dat.wide, df_flags=df_flags, 
                    cut_value=args.value, condition=args.condition, args=args)
    else:
        kpd_wide,kpd_flag = dropRows(df_wide=dat.wide, df_flags=df_flags, 
                cut_value=args.value, condition=args.condition, args=args)

    # Wide and flags
    kpd_wide.to_csv(args.outWide, sep='\t')
    kpd_flag.to_csv(args.outFlags, sep='\t')

    # Finishing script
    logger.info("Script complete.")

if __name__ == '__main__':
    #Gettign arguments from parser
    args = getOptions()

    #Stablishing logger
    logger = logging.getLogger()
    sl.setLogger(logger)

    #Change condition
    if args.condition == "0":
        args.condition="=="
    elif args.condition == "1":
        args.condition=">"
    elif args.condition == "2":
        args.condition="<"

    #Starting script
    logger.info(u"Importing data with following parameters: "\
        "\n\tWide: {0}"\
        "\n\tFlags: {1}"\
        "\n\tDesign: {2}"\
        "\n\tID: {3}"\
        "\n\toutput: {4}"\
        "\n\tVariable: {5}"\
        "\n\tCondition: {6}"\
        "\n\tValue: {7}"\
        "\n\tType Of Flag File: {8}" .format(args.input,args.flags,args.design,
                            args.uniqID,args.outWide,args.outFlags,
                            args.condition,args.value,args.flagfiletype))
    #Main script
    main(args)
    
