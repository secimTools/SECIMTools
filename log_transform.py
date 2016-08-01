#!/usr/bin/env python

# Built-in packages
import logging
import argparse
from argparse import RawDescriptionHelpFormatter

# Add-on packages
import numpy as np
import pandas as pd

# Local Packages
from interface import wideToDesign
import logger as sl


def getOptions():
    """ Function to pull in arguments """
    description = """ One-Way ANOVA """
    parser = argparse.ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--input", dest="fname", action='store', required=True, help="Input dataset in wide format.")
    parser.add_argument("--design", dest="dname", action='store', required=True, help="Design file.")
    parser.add_argument("--ID", dest="uniqID", action='store', required=True, help="Name of the column with unique identifiers.")
    parser.add_argument("--log", dest="log", action='store', choices=['log', 'log10', 'log2'], required=True, default=None, help="Group/treatment identifiers. Can be multiple names separated by a space.")
    parser.add_argument("--out", dest="oname", action='store', required=True, help="Output file name.")
    parser.add_argument("--debug", dest="debug", action='store_true', required=False, help="Add debugging log output.")
    args = parser.parse_args()
#     args = parser.parse_args(['--input', '/home/jfear/sandbox/secim/data/ST000015_AN000032_v2.txt',
#                               '--design', '/home/jfear/sandbox/secim/data/ST000015_design_v2.tsv',
#                               '--ID', 'Name',
#                               '--log', 'log',
#                               '--out', '/home/jfear/sandbox/secim/data/test.csv',
#                               '--debug'])
    return(args)


def main(args):
    # Import data
    logger.info('Importing data')
    dat = wideToDesign(args.fname, args.dname, args.uniqID)
    norm = dat.wide[dat.sampleIDs].convert_objects(convert_numeric=True)

    if args.log == 'log':
        norm = norm.apply(lambda x: np.log(x))
    elif args.log == 'log2':
        norm = norm.apply(lambda x: np.log2(x))
    elif args.log == 'log10':
        norm = norm.apply(lambda x: np.log10(x))

    # write normalized table
    norm = norm.apply(lambda x: x.round(4))
    norm.replace([np.inf, -np.inf], np.nan, inplace=True)   # Treat inf as NaN
    norm.to_csv(args.oname, sep="\t")

if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
    else:
        sl.setLogger(logger)

    main(args)
