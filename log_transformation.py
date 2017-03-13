#!/usr/bin/env python

# Built-in packages
import os
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
    parser = argparse.ArgumentParser(description=description, 
                        formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-i","--input", dest="input", action='store', 
                        required=True, help="Input dataset in wide format.")
    parser.add_argument("-d","--design", dest="design", action='store', 
                        required=True, help="Design file.")
    parser.add_argument("-id","--ID", dest="uniqID", action='store', 
                        required=True, help="Name of the column with unique"\
                        " identifiers.")
    parser.add_argument("-l", "--log", dest="log", action='store', 
                        required=True, choices=['log', 'log10', 'log2'], 
                        default=None,  help="Type of log to be used")
    parser.add_argument("-o","--out", dest="oname", action='store', 
                        required=True, help="Output file name.")
    parser.add_argument("--debug", dest="debug", action='store_true', 
                        required=False, help="Add debugging log output.")
    args = parser.parse_args()

    # Standatdize paths
    args.oname = os.path.abspath(args.oname)

    return(args)


def main(args):
    # Imput data
    dat = wideToDesign(args.input, args.design, args.uniqID, logger=logger)

    # Convert objects to numeric
    norm = dat.wide.applymap(float)

    # According to the tipe of log selected perform log transformation
    if args.log == 'log':
        logger.info(u"Running log transform with log e")
        norm = norm.apply(lambda x: np.log(x))
    elif args.log == 'log2':
        logger.info(u"Running log transform with log 2")
        norm = norm.apply(lambda x: np.log2(x))
    elif args.log == 'log10':
        logger.info(u"Running log transform with log 10")
        norm = norm.apply(lambda x: np.log10(x))

    # Round results to 4 digits
    norm = norm.apply(lambda x: x.round(4))

    # Treat inf as NaN
    norm.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Save file to CSV
    norm.to_csv(args.oname, sep="\t")
    logger.info("Finishing Script")
    
if __name__ == '__main__':
    # Command line options
    args = getOptions()

    # Set up logger
    logger = logging.getLogger()
    sl.setLogger(logger)

    # Import data
    logger.info(u"Importing data with the folowing parameters: "\
        "\n\tWide:  {0}"\
        "\n\tDesign:{1}"\
        "\n\tUniqID:{2}"\
        "\n\tLog:   {3}".\
        format(args.input,args.design,args.uniqID,args.log))

    # Runing log transformation
    main(args)
