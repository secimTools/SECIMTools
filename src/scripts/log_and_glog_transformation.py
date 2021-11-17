#!/usr/bin/env python
################################################################################
# DATE: 2017/06/15
#
# SCRIPT: log_and_glog_transformation.py
#
# VERSION: 2.2
# 
# AUTHOR: Coded by: Miguel A Ibarra (miguelib@ufl.edu) and 
# 		    Alexander Kirpich (akirpich@ufl.edu) 	 
#        Edited by: Matt Thoburn (mthoburn@ufl.edu)
# 
# DESCRIPTION: This script performs log or g-log transformation of the data.
#
################################################################################
# Impot built-in libraries
import os
import logging
import argparse
from argparse import RawDescriptionHelpFormatter

# Impot add-on libraries
import numpy as np
import pandas as pd
import math

# Impot local data libraries
from secimtools.dataManager import logger as sl
from secimtools.dataManager.interface import wideToDesign

def getOptions():
    """ Function to pull in arguments """
    description = """ Log or G-Log transformation of the data """
    parser = argparse.ArgumentParser(description=description, 
                        formatter_class=RawDescriptionHelpFormatter)
    # Tool Input
    parser.add_argument("-i","--input", dest="input", action='store', 
                        required=True, help="Input dataset in wide format.")
    parser.add_argument("-d","--design", dest="design", action='store', 
                        required=True, help="Design file.")
    parser.add_argument("-id","--uniqID", dest="uniqID", action='store', 
                        required=True, help="Name of the column with unique"\
                        " identifiers.")
    parser.add_argument("-t", "--transformation", dest="transformation", action='store', 
                        required=True, choices=['log', 'glog'], 
                        default=None,  help="Type of transformation to use (log vs glog)")
    parser.add_argument("-l", "--log_base", dest="log_base", action='store', 
                        required=True, choices=['log', 'log10', 'log2'], 
                        default=None,  help="Base of the logarithm to use")
    parser.add_argument("-la", "--lambda_value",dest="lambda_value", action='store', 
                        required=False,  default=0, help="Lambda parameter"\
                        " which is used only for G-Log transformation.")
    # Tool Output
    parser.add_argument("-o","--oname", dest="oname", action='store', 
                        required=True, help="Output file name.")
    parser.add_argument("--debug", dest="debug", action='store_true', 
                        required=False, help="Add debugging log output.")
    args = parser.parse_args()

    # Standatdize paths
    args.oname  = os.path.abspath(args.oname)
    args.input  = os.path.abspath(args.input)
    args.design = os.path.abspath(args.design)

    return(args)


def main(args):
      
    # Imput data
    dat = wideToDesign(args.input, args.design, args.uniqID, logger=logger)

    # Convert objects to numeric
    norm = dat.wide.applymap(float)

     
    # The following steps depend whether we perform log transformation or g-log transformation.

    # LOG Transformation
    if args.transformation == 'log':

       # According to the tipe of log-base selected perform log transformation
       if args.log_base == 'log':
            logger.info(u"Running Log transformation with log e")
            norm = norm.apply(lambda x: np.log(x))
       elif args.log_base == 'log2':
            logger.info(u"Running Log transformation with log 2")
            norm = norm.apply(lambda x: np.log2(x))
       elif args.log_base == 'log10':
            logger.info(u"Running Log transformation with log 10")
            norm = norm.apply(lambda x: np.log10(x))

    # G-LOG Transformation
    # Generalized log transformation formula is:  log(y + sqrt(y^2 + lambda_value))
    # It reduced to sqrt(2) rescaled version of log when lambda_value = 0
    # i.e. lambda_value == 0 implies log(y + y) = sqrt(2) * log(y)
    if args.transformation == 'glog':

       # According to the tipe of log-base selected perform log transformation
       if args.log_base == 'log':
            logger.info(u"Running G-Log transformation with log e")
            norm = np.log( norm + np.sqrt( np.square(norm) + float(args.lambda_value) ) )
       elif args.log_base == 'log2':
            logger.info(u"Running G-Log transformation with log 2")
            norm = np.log2( norm + np.sqrt( np.square(norm) + float(args.lambda_value) ) )
       elif args.log_base == 'log10':
            logger.info(u"Running G-Log transformation with log 10")
            norm = np.log10( norm + np.sqrt( np.square(norm) + float(args.lambda_value) ) )



    # Round results to 8 digits
    norm = norm.apply(lambda x: x.round(8))

    # Treat inf as NaN
    norm.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Debugging step
    # print "norm", norm

    # Save file to CSV
    norm.to_csv(args.oname, sep="\t")
    logger.info("Finishing Script")
    
if __name__ == '__main__':
    # Command line options
    args = getOptions()

    # Set up logger
    logger = logging.getLogger()
    sl.setLogger(logger)


    #Starting script
    logger.info(u"""Importing data with following parameters:
                Input: {0}
                Design: {1}
                Unique ID: {2}
                Transformation: {3}
                Log Base: {4}
                Lambda: {5}
                """.format(args.input, args.design, args.uniqID, args.transformation, args.log_base, args.lambda_value ))

    # Runing log transformation
    main(args)
