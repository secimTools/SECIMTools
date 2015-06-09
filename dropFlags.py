#!/usr/bin/env python

# Author: Jonathan Poisson | poissonj@ufl.edu

# Built-in packages
import argparse
from argparse import RawDescriptionHelpFormatter

# Add-on packages
import pandas as pd

# Local Packages
from interface import wideToDesign


def getOptions():
    """ Function to pull in arguments """
    parser = argparse.ArgumentParser(description="", formatter_class=RawDescriptionHelpFormatter)

    group1 = parser.add_argument_group(title='Standard input', description='Standard input for SECIM tools.')
    group1.add_argument("--input", dest="fname", action='store', required=True, help="Input dataset in wide format.")
    group1.add_argument("--design", dest="dname", action='store', required=True, help="Design file.")
    group1.add_argument("--ID", dest="uniqID", action='store', required=True,
                        help="Name of the column with unique identifiers.")
    group1.add_argument("--group", dest="group", action='store', default=False, required=False,
                        help="Group/treatment identifier in design file [Optional].")
    group1.add_argument("--flags", dest="flagFile", action='store', required=True, help="Input flag file")


    group2 = parser.add_argument_group(title='Required input', description='Additional required input for this tool.')
    group2.add_argument("--cutoff", dest="cutoff", action='store', type=int, required=True,
                        help="Cutoff value for dropping. Any flag sum value greater than this number will be dropped")
    group1.add_argument("--wideOut", dest="wideOut", action='store', required=True, help="Output Wide Dataset")
    group1.add_argument("--designOut", dest="designOut", action='store', required=False,
                        help="Output Design Dataset. Only required when columns are wanted to be dropped.")

    group3 = parser.add_mutually_exclusive_group(required=True)
    group3.add_argument("--row", dest="dropRow", action="store_true", help="Drop rows.")
    group3.add_argument("--column", dest="dropColumn", action="store_true", help="Drop columns.")

    args = parser.parse_args()
    return args


def dropRows(df_wide, df_flags, cutoffValue, args):
    """ Drop rows in a log file based on its flag file and the specified flag values to keep.

    Arguments:

        :type df_wide: pandas.DataFrame
        :param df: A data frame in wide format

        :type df_wide: pandas.DataFrame
        :param df: A data frame in design format

        :type df_flags: pandas.DataFrame
        :param df: A data frame of flag values corresponding to the wide file

        :param integer cutoffValue: The int value of the cutoff for which flags to keep.
            All flag sums per row greater than this number will cause the row to be deleted

    Returns: Wide DataFrame with dropped rows

    """


    # This method will create a sum column and then create a mask to delete the flagged values from the original data

    # Create a sum column and add to the end of the flag file
    sumColumn = df_flags.sum(numeric_only=True, axis=1)
    df_flags['sum'] = sumColumn

    # Only keep the rows in the original data that the user specified
    df_flags = df_flags.loc[df_flags['sum'] == cutoffValue]

    # Create a mask over the original data to determine what to delete
    mask = df_wide.index.isin(df_flags.index)

    # Use mask to drop values form original data
    df_wide = df_wide[mask]

    # What should I name them?
    df_wide.to_csv(args.wideOut)


def dropColumns(df_wide, df_design, df_flags, cutoffValue, args):
    """ Drop columns in both the wide and design files based on the sampleID's flag sums.

    Arguments:

        :type df_wide: pandas.DataFrame
        :param df_wide: A data frame in wide format

        :type df_design: pandas.DataFrame
        :param df_design: A data frame in design format

        :type df_flags: pandas.DataFrame
        :param df_flags: A data frame of flag values corresponding to the wide and design files

        :param int cutoffValue: The integer value of the cutoff for which flags to keep.
            All flag sums per column greater than this number will cause that corresponding column to be deleted

    Returns: Both wide and design files with dropped columns
    """

    # Sum the Columns
    sumRow = {col: df_flags[col].sum() for col in df_flags}

    # Turn the sums into a DataFrame with one row with an index of 'Total'
    sumDf = pd.DataFrame(sumRow, index=['Total'])

    # Append the rows to the flag data
    df_flags = df_flags.append(sumDf)

    # Only keep the columns in the original data that the user specified

    # Get the names of the compounds from the df_wide file into a list to loop through. These column values are
    # located as the column values in the flags DataFrame
    sampleIDList = df_wide.columns

    # Delete from the df_flags the totals that are greater than the cutoff value
    for sampleID in sampleIDList:
        if df_flags[sampleID]['Total'] > cutoffValue:  # If each columns total is greater than one of the flag values
            del df_flags[sampleID]

    # Now that the df_flags has dropped its sampleID columns, match the df_with's columns with the flags by comparing
    # the columns and droping the columns from df_wide that are not in df_flags
    for wideColumn in df_wide.columns:
        if wideColumn not in df_flags.columns:
            del df_wide[wideColumn]

    # Complete same action with the df_design file so all of the sampleID's are the same in all 3 files
    for designIndex in df_design.index:
        if designIndex not in df_flags.columns:
            # df_design = df_design.loc[designIndex]
            df_design = df_design.drop(designIndex, axis=0)


    # Sort tables before exporting
    df_wide = df_wide.sort(axis=1)
    df_flags = df_flags.sort(axis=1)
    df_design = df_design.sort(axis=0)

    # Export the files
    df_wide.to_csv(args.wideOut)
    df_design.to_csv(args.designOut)


def main(args):
    # Execute wideToDesign to make all data uniform
    formatted_data = wideToDesign(wide=args.fname, design=args.dname, uniqID=args.uniqID, group=args.group)

    # Convert flag file to DataFrame
    df_flags = pd.DataFrame.from_csv(args.flagFile, sep='\t')

    # If the user specified rows, run dropRows
    if args.dropRow:
        dropRows(df_wide=formatted_data.wide, df_flags=df_flags, cutoffValue=args.cutoff, args=args)

    # If the user specified columns, run dropColumns
    else:  # (if args.dropColumn:)
        dropColumns(df_wide=formatted_data.wide, df_design=formatted_data.design, df_flags=df_flags,
                    cutoffValue=args.cutoff, args=args)


if __name__ == '__main__':
    # Command line options
    args = getOptions()
    # Run the main function with data from wideToDesign and the flag file
    main(args)
