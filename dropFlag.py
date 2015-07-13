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
    description = """
    This tool takes in a Wide formatted dataset, a Design
    formatted dataset and corresponding flag values. With the inputted user
    cutoff value the flag values are summed and the sums that are greater than
    the cutoff value are dropped. If rows are the specified attribute to drop
    then the return will be a new Wide file but if columns are dropped then a
    new Wide file and Design file will be created because changes will have
    occurred in each file."""

    parser = argparse.ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)

    group1 = parser.add_argument_group(title='Standard input', description='Standard input for SECIM tools.')
    group1.add_argument("--input", dest="fname", action='store', required=True, help="Input dataset in wide format.")
    group1.add_argument("--design", dest="dname", action='store', required=True, help="Design file.")
    group1.add_argument("--ID", dest="uniqID", action='store', required=True,
                        help="Name of the column with unique identifiers.")
    group1.add_argument("--group", dest="group", action='store', default=False, required=False,
                        help="Group/treatment identifier in design file [Optional].")
    group1.add_argument("--flags", dest="flagFile", action='store', required=True, help="Input flag file")

    group2 = parser.add_argument_group(title='Required input', description='Additional required input for this tool.')
    group2.add_argument("--cutoff", dest="cutoff", action='store', type=float, required=True, default=.5,
                        help="Cutoff value for dropping. Drop row if the porportion is less than this value. The default cutoff is .5")
    group2.add_argument("--wideOut", dest="wideOut", action='store', required=True, help="Output Wide Dataset")
    group2.add_argument("--designOut", dest="designOut", action='store', required=False,
                        help="Output Design Dataset. Only required when columns are wanted to be dropped.")

    group3 = parser.add_mutually_exclusive_group(required=True)
    group3.add_argument("--row", dest="dropRow", action="store_true", help="Drop rows.")
    group3.add_argument("--column", dest="dropColumn", action="store_true", help="Drop columns.")

    args = parser.parse_args()

    # Check if cutoff value is a porportion
    if args.cutoff > 1 or args.cutoff < 0:
        parser.error("Cutoff value needs to be a porportion")

    return args


def dropRows(df_wide, df_flags, cutoffValue, args):
    """ Drop rows in a log file based on its flag file and the specified flag values to keep.

    :Arguments:
        :type df_wide: pandas.DataFrame
        :param df: A data frame in wide format

        :type df_wide: pandas.DataFrame
        :param df: A data frame in design format

        :type df_flags: pandas.DataFrame
        :param df: A data frame of flag values corresponding to the wide file

        :param integer cutoffValue: The int value of the cutoff for which flags to keep.
            All flag sums per row greater than this number will cause the row to be deleted

        :type args: argparse.ArgumentParser
        :param args: Command line arguments.

    :Returns:
        :rtype: pandas.DataFrame
        :returns: Updates the wide DataFrame with dropped rows and writes to a
            TSV.

    """
    # Create a sum column and add to the end of the flag file
    meanColumn = df_flags.mean(numeric_only=True, axis=1, skipna=True)
    df_flags['mean'] = meanColumn

    # Only keep the rows in the original data that the user specified
    df_flags = df_flags.loc[df_flags['mean'] < cutoffValue]

    # Create a mask over the original data to determine what to delete
    mask = df_wide.index.isin(df_flags.index)

    # Use mask to drop values form original data
    df_wide = df_wide[mask]

    # Export
    df_wide.to_csv(args.wideOut, sep='\t')


def dropColumns(df_wide, df_design, df_flags, cutoffValue, args):
    """ Drop columns in both the wide and design files based on the sampleID's flag sums.

    :Arguments:

        :type df_wide: pandas.DataFrame
        :param df_wide: A data frame in wide format

        :type df_design: pandas.DataFrame
        :param df_design: A data frame in design format

        :type df_flags: pandas.DataFrame
        :param df_flags: A data frame of flag values corresponding to the wide and design files

        :param int cutoffValue: The integer value of the cutoff for which flags to keep.
            All flag sums per column greater than this number will cause that corresponding column to be deleted

        :type args: argparse.ArgumentParser
        :param args: Command line arguments.

    :Returns:
        :rtype: pandas.DataFrame
        :returns: Both wide and design files with dropped columns
    """

    # Mean the Columns and create new row at the bottom named Mean
    df_flags.loc['Mean', :] = df_flags.mean(axis=0)

    # Create list of sampleIDs that are greater than or equal to the cutoff
    df_flaggedIDs = df_flags.columns[df_flags.loc['Mean', :] >= cutoffValue]

    # Create list of sampleIDs to keep in wide DataFrame.
    #
    # NOTE: the ~ inverts boolean values.
    #
    # Take the list of values to drop 'df_flaggedIDs', see if df_wide column
    # headers are in this list. Invert the boolean to keep headers that are not
    # in the list.
    keepSampleIDs = df_wide.columns[~df_wide.columns.isin(df_flaggedIDs)]

    # Pull out columns from wide
    df_wide = df_wide[keepSampleIDs]

    # Pull out rows from design file
    df_design = df_design[df_design.index.isin(keepSampleIDs)]

    # Sort tables before exporting
    df_wide = df_wide.sort(axis=1)
    df_design = df_design.sort(axis=0)

    # Export the files
    df_wide.to_csv(args.wideOut, sep='\t')
    df_design.to_csv(args.designOut, sep='\t')


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

