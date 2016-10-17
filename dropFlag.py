#!/usr/bin/env python
################################################################################
# SCRIPT: dropFlag_v2.1.py
# 
# LAST VERSION: Includes support to select the type of drop to perform instead
#               of automatically decide. Also includes better recognition 
#               between drop by string or by number. (MIA)
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
# NOTES: Still needs to be added functionality to select index in flag file.
################################################################################

#Standard Libraries
import re
import copy
import logging
import argparse
from argparse import RawDescriptionHelpFormatter

#Add-on Libraries
import pandas as pd

#Local Libraries
import logger as sl
from interface import wideToDesign


"""Function to pull arguments"""
def getOptions():
    parser = argparse.ArgumentParser(description="Drops rows or columns given" \
                                    "an specific cut value and condition.")
    
    required = parser.add_argument_group(title='Required Input', 
                                        description="Requiered input to the "\
                                        "program.")
    required.add_argument('-i',"--input",dest="input", action="store",
                        required=True,help="Input dataset in wide format.")
    required.add_argument('-d',"--design",dest="designFile", action="store",
                        required=True, help="Design file.")
    required.add_argument('-f',"--flags",dest="flagFile", action="store",
                        required=True,help="Flag file.")
    required.add_argument('-id',"--ID",dest="uniqID",action="store",
                        required=True,help="Name of the column with unique "\
                        "identifiers.")
    required.add_argument('-fid',"--flagUniqID",dest="flagUniqID",action="store",
                        required=False, default=False,help="Name of the column "\
                        "with unique identifiers in the flag files.")
    required.add_argument('-fd',"--flagDrop",dest="flagDrop",action='store',
                        required=True, help="Name of the flag/field you want to"\
                        " access.")

    output = parser.add_argument_group(title='Output', description="Output of "\
                                        "the script.")
    output.add_argument('-ow',"--outWide",dest="outWide",action="store",required=True,
                        help="Output file without the Drops.")
    output.add_argument('-of',"--outFlags",dest="outFlags",action="store",
                        required=True,help="Output file for Drops.")

    optional = parser.add_argument_group(title="Optional Input", 
                                    description="Optional Input to the program.")
    optional.add_argument('-fft',"--flagfiletype",dest="flagfiletype",action='store',
                        required=True, default="row", choices=["row","column"],
                        help="Type of flag file")                                   
    optional.add_argument('-val',"--value",dest="value",action='store',
                        required=False, default="1",help="Cut Value")
    optional.add_argument('-con',"--condition",dest="condition",action='store',
                        required=False, default="0",help="Condition for the cut" \
                        "where 0=Equal to, 1=Greater than and 2=less than.")

    args = parser.parse_args()
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

    #Export wide
    df_wide_keeped.to_csv(args.outWide, sep='\t')

    #Export flags
    df_flags_keeped.to_csv(args.outFlags, sep='\t')

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
            samples_to_drop = df_flags.index[df_flags[args.flagDrop]<cut_value]
        elif condition == '<':
            samples_to_drop = df_flags.index[df_flags[args.flagDrop]>cut_value]
        elif condition == '==':
            samples_to_drop = df_flags.index[df_flags[args.flagDrop]!=cut_value]
        else:
            logger.error(u'The {0} is not supported by the program, please use <,== or >'.format(condition))
            quit()
    else:
        cut_value = str(cut_value)
        if condition == '==':
            samples_to_drop = df_flags.index[df_flags[args.flagDrop]!=cut_value]
        else:
            logger.error(u'The {0} conditional is not supported for string flags, please use =='.format(condition))
            quit()

    dropped_flags = df_flags.T[samples_to_drop].T

    #Output 
    df_wide.to_csv(args.outWide, columns=samples_to_drop, sep='\t')
    dropped_flags.to_csv(args.outFlags, sep='\t')

def main(args):
    # Execute wideToDesign to make all data uniform
    formatted_data = wideToDesign(wide=args.input, design=args.designFile, 
                                uniqID=args.uniqID)    

    # Convert flag file to DataFrame
    df_flags = pd.read_table(args.flagFile)

    # Select index on flag file
    # If user gives a flagUniqID the override automatic response
    if args.flagUniqID:
        df_flags.set_index(args.flagUniqID, inplace=True)
    # If not flagUniqID then decides a default one based on type of drop if 
    # column use 'sampleID'.
    elif args.flagfiletype=="column":
        df_flags.set_index("sampleID", inplace=True)
    # If not flagUniqID then decides a default one based on type of drop if 
    # row use 'rowID'.
    elif args.flagfiletype=="row":
        df_flags.set_index("rowID", inplace=True)
    # Send error if none 
    else:
        logger.error("Not flagUniqID provided")

    # Drop either rows or columns
    if args.flagfiletype=="column":
        logger.info("Runing drop flags by Column")

        # Running dropColumns routine
        dropColumns(df_wide=formatted_data.wide, df_flags=df_flags, 
                    cut_value=args.value, condition=args.condition, args=args)

    else:
        logger.info("Runing drop flags by Row")

        # Running dropRows routine
        dropRows(df_wide=formatted_data.wide, df_flags=df_flags, 
                cut_value=args.value, condition=args.condition, args=args)

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
        "\n\tType Of Flag File: {8}" .format(args.input,args.flagFile,args.designFile,
                            args.uniqID,args.outWide,args.outFlags,
                            args.condition,args.value,args.flagfiletype))
    #Main script
    main(args)
    