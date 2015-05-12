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
    group1.add_argument("--group", dest="group", action='store', default=False, required=False, help="Group/treatment identifier in design file [Optional].")

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


def main(args):
    # Import data
    logger.info('Importing Data')
    dat = wideToDesign(args.fname, args.dname, args.uniqID, args.group)
    df = dat.wide.copy()

    # Select Method
    if args.method == 'svd':
        scale = center = ri.FALSE
        if args.svd == 'both':
            scale = center = ri.TRUE
        elif args.svd == 'center':
            center = ri.TRUE
        elif args.svd == 'scale':
            scale = ri.TRUE

    # Set up R commands
    r = ro.r
    rNA = r['na.exclude']
    rprincomp = r.princomp
    rsummary = r.summary

    # Run PCA
    try:
        if args.method == "cor":
            pc = rprincomp(rNA(df), cor=ri.TRUE)
        elif args.method == "cov":
            pc = rprincomp(rNA(df), cor=ri.FALSE)
        elif args.method == "svd":
            pc = rprincomp(rNA(df), center=center, scale=scale)
    except:
        logger.error('R principal components failed to run, check rpy2.')

    # Summarize PCA
    summary = rsummary(pc, loadings=ri.TRUE)
    summary.rx2('sdev')
    sdev = summary.rx2('sdev')
    loadings = summary.rx2('loadings')
    scores = summary.rx2('scores')

    # Create output data sets
    components = pd.DataFrame({'#Component': [x.split('.')[1] for x in sdev.names]})
    sd = pd.DataFrame({'#Std. deviation': [x for x in sdev]})
    prop = sd.apply(lambda x: x**2 / np.sum(sd**2), axis=1)
    prop.columns = ['#Proportion of variance explained', ]

    load_index = pd.Series([x for x in loadings.names[0]], name='#Loadings')
    load_df = pandas2ri.ri2py_dataframe(loadings)
    load_df.index = load_index
    load_out = pd.concat([components.T, sd.T, prop.T, load_df])

    score_index = pd.DataFrame({'#Scores': [x for x in scores.names[0]]})
    score_df = pandas2ri.ri2py_dataframe(scores)
    score_df.index = score_index
    score_out = pd.concat([components.T, sd.T, prop.T, score_df])

    # Write output
    load_out.to_csv(args.lname, sep='\t', header=False)
    score_out.to_csv(args.sname, sep='\t', header=False)


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
        global DEBUG
        DEBUG = True
    else:
        sl.setLogger(logger)
        global DEBUG
        DEBUG = False

    main(args)
