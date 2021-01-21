#!/usr/bin/env python
####################################################################################################
# SCRIPT: compare_flags.py
# AUTHORS: Miguel Ibarra Arellano <miguelib@ufl.edu>
#          Oleksandr Moskalenko <om@rc.ufl.edu>
#
# DESCRIPTION: This script takes a Wide format file (wide) or a flag file, and creates a cross
# tabulated file (a file with the frequency between two selected flags for the flag file).
#
# INPUT:
#        A wide format file or a flag file [File]
#        The name of two flags  (as it appears in the flag file) to be assessed.
#
# OUTPUT:
#        A cross tabulated file with the frequencies of those two selected flags.
#
###################################################################################################

import logging
import argparse
import re
import pandas as pd
import numpy as np
from argparse import RawDescriptionHelpFormatter
from secimtools.dataManager import logger as sl


def getOptions():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Samples a wide formated file or a flag file"
    )

    required = parser.add_argument_group(
        title="Required Input", description="Requiered input to the program."
    )
    required.add_argument(
        "-i",
        "--input",
        dest="inputFile",
        action="store",
        required=True,
        help="Input dataset in wide format.",
    )
    required.add_argument(
        "-o",
        "--output",
        dest="output",
        action="store",
        required=True,
        help="Output file (Cross tab format).",
    )
    required.add_argument(
        "-f1",
        "--flag1",
        dest="flag1",
        action="store",
        required=True,
        help="Flag 1 to create cross tab",
    )
    required.add_argument(
        "-f2",
        "--flag2",
        dest="flag2",
        action="store",
        required=True,
        help="Flag 2 to create cross tab",
    )

    args = parser.parse_args()
    return args


def flagCounter(df_wide, flag1, flag2):
    """
    Description:
            Calculates the frequency among the flags
    Input:
            df_wide:    (Pandas DataFrame) with the flag file
            flag1:        (String) Name of the first flag.
            flag2:        (String) Name of the second  flag.
    Output:
            crossTab:    (dictionary(dictionary(int|str))) with the flag combinations and its
            respective frequency 
    """
    # Subseting dataFrame
    df_flags = df_wide.loc[:, [flag1, flag2]]

    # Get all diferent values for a flag 1
    flag_list1 = list(set(df_wide[flag1].tolist()))
    flag_list2 = list(set(df_wide[flag2].tolist()))

    # Calculating flag counts
    crossTab = {}
    for flag_f1 in flag_list1:
        try:
            df_droppedflags_f1 = df_flags[df_flags[flag1] == flag_f1]
        except TypeError:
            df_droppedflags_f1 = df_flags[df_flags[flag1] == int(flag_f1)]
        crossTab[flag1 + "_" + str(flag_f1)] = {}
        for flag_f2 in flag_list2:
            try:
                df_droppedflags_f2 = df_droppedflags_f1[
                    df_droppedflags_f1[flag2] == flag_f2
                ]
            except TypeError:
                df_droppedflags_f2 = df_droppedflags_f1[
                    df_droppedflags_f1[flag2] == str(flag_f2)
                ]
            crossTab[flag1 + "_" + str(flag_f1)][(flag2 + "_" + str(flag_f2))] = len(
                df_droppedflags_f2.index
            )
    return crossTab


def main(args):
    logger.info(
        u"Importing data with following parameters: \n\tWide: {0}\n\tOutput: {1}\n\tFlag 1: {2}\n\tFlag 2: {3}".format(
            args.inputFile, args.output, args.flag1, args.flag2
        )
    )

    # Getting a data frame from the
    df_wide = pd.read_csv(args.inputFile, sep="\t")

    # Counting flag combinations
    crossTab = flagCounter(df_wide, args.flag1, args.flag2)

    # Creating dataFrame from dictionary
    df_crossTab = pd.DataFrame(crossTab)

    # Exporting file
    df_crossTab.to_csv(args.output, sep="\t")
    logger.info("Script complete.")


if __name__ == "__main__":
    args = getOptions()
    logger = logging.getLogger()
    sl.setLogger(logger)
    main(args)
