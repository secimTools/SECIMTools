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
    parser.add_argument("--group", dest="group", action='store', required=True, help="Group/treatment identifier in design file.")
    parser.add_argument("--std", dest="std", action='store', choices=['STD', 'MEAN'], required=True, default=None, help="Group/treatment identifiers. Can be multiple names separated by a space.")
    parser.add_argument("--out", dest="oname", action='store', required=True, help="Output file name.")
    parser.add_argument("--debug", dest="debug", action='store_true', required=False, help="Add debugging log output.")
    args = parser.parse_args()
#     args = parser.parse_args(['--input', '/home/jfear/sandbox/secim/data/ST000015_AN000032_v2.tsv',
#                               '--design', '/home/jfear/sandbox/secim/data/ST000015_design_v2.tsv',
#                               '--ID', 'Name',
#                               '--group', 'treatment',
#                               '--std', 'STD',
#                               '--out', '/home/jfear/sandbox/secim/data/test.csv',
#                               '--debug'])
    return(args)


def sasSTD(dat):
    """ Function standardize using SAS STD procedure """
    # Get transposed table
    trans = dat.transpose()

    # group by treatment
    grp = trans.groupby(dat.group)

    # Calculate group means and std
    meanMean = grp.mean().mean()
    stdMean = grp.mean().std()

    # Standardize using sas STD method
    standard = grp.transform(lambda x: (x - meanMean[x.name]) / stdMean[x.name])

    return standard


def sasMEAN(dat):
    """ Function standardize using SAS STD procedure """
    # Get transposed table
    trans = dat.transpose()

    # group by treatment
    grp = trans.groupby(dat.group)

    # Calculate group means and std
    meanMean = grp.mean().mean()

    # Standardize using sas MEAN method
    standard = grp.transform(lambda x: x - meanMean[x.name])

    return standard


def main(args):
    # Import data
    logger.info('Importing data')
    dat = wideToDesign(args.fname, args.dname, args.uniqID, args.group)

    # Standardize the data
    if args.std == 'STD':
        logger.info('Calculating STD standardization')
        dat.sdat = sasSTD(dat)
    elif args.std == 'MEAN':
        logger.info('Calculating MEAN standardization')
        dat.sdat = sasMEAN(dat)

    # write standardized table
    standard = dat.sdat.convert_objects(convert_numeric=True)
    standard = standard.apply(lambda x: x.round(4))
    standardT = standard.T
    standardT.index.name = dat.uniqID
    standardT.to_csv(args.oname, sep="\t")


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
    else:
        sl.setLogger(logger)

    main(args)
