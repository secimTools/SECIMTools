#!/usr/bin/env python
# Author: Jonathan Poisson | PoissonJ@ufl.edu


# Built-in packages
import argparse


# Local packages
from interface import wideToDesign


def getOptions():
    """ Function to pull in arguments """

    description = ""

    parser = argparse.ArgumentParser(description=description)

    group1 = parser.add_argument_group(title='Standard input', description='Standard input for SECIM tools.')
    group1.add_argument("--input", dest="fname", action='store', required=True, help="Input dataset in wide format.")
    group1.add_argument("--design", dest="dname", action='store', required=True, help="Design file.")
    group1.add_argument("--ID", dest="uniqID", action='store', required=True, help="Name of the column with unique identifiers.")
    group1.add_argument("--wideOut", dest="wideOut", action='store', required=True, help="File destination of the cleaned wide file")
    group1.add_argument("--designOut", dest="designOut", action='store', required=True, help="File destination of the cleaned design file")

    args = parser.parse_args()

    return (args)


def main(args):

    # Import data with clean string as true
    df_cleanedData = wideToDesign(wide=args.fname, design=args.dname,
                                  uniqID=args.uniqID, clean_string=True)

    # Export cleaned data
    df_cleanedData.wide.to_csv(args.wideOut, sep="\t")
    df_cleanedData.design.to_csv(args.designOut, sep="\t")


if __name__ == '__main__':

    # Command line options
    args = getOptions()
    main(args)
