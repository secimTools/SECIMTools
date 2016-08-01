#!/usr/bin/env python

# Built-in packages
from __future__ import division
import logging
import argparse
from argparse import RawDescriptionHelpFormatter

# Add-on packages
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Local Packages
from interface import wideToDesign
import logger as sl


def getOptions(myopts=None):
    """ Function to pull in arguments """
    description = """  """
    parser = argparse.ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)
    group1 = parser.add_argument_group(title='Standard input', description='Standard input for SECIM tools.')
    group1.add_argument("--input", dest="fname", action='store', required=True, help="Input dataset in wide format.")
    group1.add_argument("--design", dest="dname", action='store', required=True, help="Design file.")
    group1.add_argument("--ID", dest="uniqID", action='store', required=True, help="Name of the column with unique identifiers.")

    group2 = parser.add_argument_group(title='Required output')
    group2.add_argument("--load_out", dest="lname", action='store', required=True, help="Name of output file to store loadings. TSV format.")
    group2.add_argument("--score_out", dest="sname", action='store', required=True, help="Name of output file to store scores. TSV format.")

    group4 = parser.add_argument_group(title='Development Settings')
    group4.add_argument("--debug", dest="debug", action='store_true', required=False, help="Add debugging log output.")

    args = parser.parse_args()

    return(args)


def main(args):
    # Import data
    logger.info('Importing Data')
    dat = wideToDesign(args.fname, args.dname, args.uniqID)
    df_wide = dat.wide[dat.sampleIDs].copy()

    # Drop Missing
    if np.isnan(df_wide.values).any():
        nRows = df_wide.shape[0]        # Number of rows before dropping missing
        df_wide.dropna(inplace=True)    # Drop missing rows in place
        nRowsNoMiss = df_wide.shape[0]  # Number of rows after dropping missing
        logger.warn('{} rows were dropped because of missing values.'.format(nRows - nRowsNoMiss))

    # Run PCA
    # Initialize PCA class with default values
    pca = PCA()

    # Fit PCA
    scores = pca.fit_transform(df_wide)

    # Get loadings
    loadings = pca.components_

    # Get additional information
    sd = scores.std(axis=0)
    propVar = pca.explained_variance_ratio_
    cumPropVar = propVar.cumsum()

    # Create header block for output. I am replicating output from R, which
    # includes additional information (above) at the top of the output file.
    labels = np.array(['#Std. deviation', '#Proportion of variance explained', '#Cumulative proportion of variance explained'])
    blockDat = np.vstack([sd, propVar, cumPropVar])
    block = np.column_stack([labels, blockDat])

    # Create header for output
    header = np.array(['PC{}'.format(x + 1) for x in range(loadings.shape[1])])
    compoundIndex = np.hstack([df_wide.index.name, df_wide.index])
    sampleIndex = np.hstack(['sampleID', df_wide.columns])

    # Create loadings output
    loadHead = np.vstack([header, loadings])
    loadIndex = np.column_stack([sampleIndex, loadHead])
    loadOut = np.vstack([block, loadIndex])

    # Create scores output
    scoreHead = np.vstack([header, scores])
    scoreIndex = np.column_stack([compoundIndex, scoreHead])
    scoreOut = np.vstack([block, scoreIndex])

    # Save output
    np.savetxt(args.lname, loadOut, fmt='%s', delimiter='\t')
    np.savetxt(args.sname, scoreOut, fmt='%s', delimiter='\t')


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
    else:
        sl.setLogger(logger)

    main(args)
