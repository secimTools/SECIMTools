#!/usr/bin/env python

# Built-in packages
from __future__ import division
import logging
import argparse
from argparse import RawDescriptionHelpFormatter

# Add-on packages
import pandas as pd
import numpy as np
import rpy2.robjects as ro
import rpy2.rinterface as ri
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()

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

    group2 = parser.add_argument_group(title='Required input', description='Additional required input for this tool.')
    group2.add_argument("--method", dest="method", action='store', choices=['cov', 'cor', 'svd'], required=True, help="Name of the output PDF for Bland-Altman plots.")
    group2.add_argument("--svd_options", dest="svd", action='store', choices=['both', 'center', 'scale'], required=False, help="Name of the output PDF for Bland-Altman plots.")

    group3 = parser.add_argument_group(title='Required output')
    group3.add_argument("--load_out", dest="lname", action='store', required=True, help="Name of output file to store loadings.")
    group3.add_argument("--score_out", dest="sname", action='store', required=True, help="Name of output file to store scores.")

    group4 = parser.add_argument_group(title='Development Settings')
    group4.add_argument("--debug", dest="debug", action='store_true', required=False, help="Add debugging log output.")

    args = parser.parse_args()

    return(args)


def cleanR(x):
    """ Helper function to remove X's if variable names start with x. """
    if x.startswith('X'):
        return x[1:]
    else:
        return x


def princomp(data, args, cor=True):
    stats = importr('stats')

    # Run PCA
    pc = stats.princomp(stats.na_exclude(data), cor=cor)

    # Summarize PCA
    summary = stats.summary_princomp(pc, loadings=True)
    sdev = summary.rx2('sdev')
    loadings = summary.rx2('loadings')
    scores = summary.rx2('scores')

    # Create output data sets
    components = pd.DataFrame({'#Component': ['PC'+str(x) for x in range(1, len(sdev) + 1)]})
    sd = pd.DataFrame({'#Std. deviation': [x for x in sdev]})
    prop = sd.apply(lambda x: x**2 / np.sum(sd**2), axis=1)
    prop.columns = ['#Proportion of variance explained', ]

    load_index = pd.Series([x for x in loadings.names[0]], name='#Loadings')
    if load_index.str.startswith('X', na=False).all():
        #TODO: If sampleIDs are int, then R returns sample ID as X + int. This hack removes 'X' for now.
        load_index = load_index.apply(cleanR)
    load_df = pandas2ri.ri2py_dataframe(loadings)
    load_df.index = load_index
    load_out = pd.concat([components.T, sd.T, prop.T, load_df])

    score_index = pd.DataFrame({'#Scores': [x for x in scores.names[0]]})
    score_df = pandas2ri.ri2py_dataframe(scores)
    score_df.index = [str(x) for x in scores.names[0]]
    score_out = pd.concat([components.T, sd.T, prop.T, score_df])

    # Write output
    load_out.to_csv(args.lname, sep='\t', header=False)
    score_out.to_csv(args.sname, sep='\t', header=False)

def svd(data, args, svdOpt):
    stats = importr('stats')

    if svdOpt == 'both':
        myopts = {'scale.': True, 'center': True}
    elif svdOpt == 'center':
        myopts = {'scale.': False, 'center': True}
    elif svdOpt == 'scale.':
        myopts = {'scale.': True, 'center': False}
    else:
        myopts = {'scale.': False, 'center': False}

    pc = stats.prcomp(stats.na_exclude(data), **myopts)
    summary = stats.summary_prcomp(pc)
    sdev = summary.rx2('sdev')
    loadings = summary.rx2('rotation')

    # Create output data sets
    components = pd.DataFrame({'#Component': ['PC'+str(x) for x in range(1, len(sdev) + 1)]})
    header = pandas2ri.ri2py_dataframe(summary.rx2('importance'))[:-1]      # drop last row
    header.index = ['#Std. deviation', '#Proportion of variance explained']

    load_index = pd.Series([x for x in loadings.names[0]], name='#Loadings')
    load_df = pandas2ri.ri2py_dataframe(loadings)
    load_df.index = load_index
    load_out = pd.concat([components.T, header, load_df])

    # Write output
    load_out.to_csv(args.lname, sep='\t', header=False)

def main(args):
    # Import data
    logger.info('Importing Data')
    dat = wideToDesign(args.fname, args.dname, args.uniqID, clean_string=True)
    df = dat.wide.copy()

    # Run PCA
    if args.method == "cor":
        princomp(df, args, cor=True)
    elif args.method == "cov":
        princomp(df, args, cor=False)
    elif args.method == "svd":
        svd(df, args, args.svd)

if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
    else:
        sl.setLogger(logger)

    main(args)
