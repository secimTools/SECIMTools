#!/usr/bin/env python
###############################################################################################################
# SCRIPT: remove_user_specified_feature_samples.py
#
# AUTHOR: Gavin Gamble and Alison Morse (ammorse@ufl.edu)
#
# DESCRIPTION: This script takes a Wide format file and a design file.  The user enters rows/features and-or
# columns/samples.  The script drops the user-specified rows and/or columns.
#
# OUTPUT:
#    Wide file without the user specified rows/columns
#
###############################################################################################################
# Import built-in libraries

# Import add-on libraries
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
    parser = argparse.ArgumentParser(description="Removes user-specified rows or columns from the data")
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
    # Tool input
    parser.add_argument('-r', "--row", dest="row", nargs='*', action="store", required=True,
                        help="Row/feature to be dropped - use rowID (can be left empty)")
    parser.add_argument('-c', "--col", dest="col", nargs='*', action="store", required=True,
                        help="Column/sample to be dropped - use column header(can be left empty)")
    # Tool Output
    output = parser.add_argument_group(title='Output',
                        description="Output of the script.")
    output.add_argument('-ow',"--outWide",dest="outWide",action="store",required=True,
                        help="Output file without specified rows/columns.")
    args = parser.parse_args()

    # Standardize paths
    args.input    = os.path.abspath(args.input)
    args.design   = os.path.abspath(args.design)
    args.outWide  = os.path.abspath(args.outWide)

    return(args);

#Making the wide file a dataframe, then removing specified rows and/or columns
def dropRowCol(df_wide, args):
    """Drop rows and / or columns in a wide file based on list(s) provided by user.

    :Arguments:
        :type df_wide: pandas.DataFrame
        :param df: A data frame in wide format

        :type rowID: basestring
        :param args: rows to drop

        :type colID: string
        :param args: Cols to drop

        :type args: argparse.ArgumentParser
        :param args: Command line arguments.

    :Returns:
        :rtype: pandas.DataFrame
        :returns: updates the wide DataFrame where indicated rows and cols are dropped and writes to tsv
    """
    # drop rows
    df_row_UPD = df_wide.drop(args.row)

    # now drop cols
    cols_not_in_df = [col for col in args.col if col not in df_row_UPD.columns]
    print("Columns not in data: {}".format(cols_not_in_df))
    cols_in_df = [col for col in args.col if col in df_row_UPD.columns]
    df_col_UPD = df_row_UPD.drop(cols_in_df, axis=1)

    return df_col_UPD

def main(args):
    # import data with interface
    dat = wideToDesign(wide=args.input, design=args.design, uniqID=args.uniqID, logger=logger)

    kpd_wide=dropRowCol(dat.wide, args=args)

    # output new wide dataset
    kpd_wide.to_csv(args.outWide, sep='\t')

if __name__ == '__main__':
    #Gettign arguments from parser
    args = getOptions()

    #Stablishing logger
    logger = logging.getLogger()
    sl.setLogger(logger)

    #Starting script
    logger.info(u"Importing data with following parameters: "\
        "\n\tWide: {0}"\
        "\n\tDesign: {1}"\
        "\n\tID: {2}"\
        "\n\toutput: {3}"\
        "\n\tRowID: {4}"\
        "\n\tColID: {5}" .format(args.input,args.design,args.uniqID,args.outWide,args.row,args.col))
    #Main script
    main(args)

