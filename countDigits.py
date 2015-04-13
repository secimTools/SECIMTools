
#!/usr/bin/env python

# Built-in packages
import logging
import argparse
from argparse import RawDescriptionHelpFormatter

# Add-on packages
import numpy as np

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
    parser.add_argument("--out", dest="oname", action='store', required=True, help="Output file name.")
    parser.add_argument("--debug", dest="debug", action='store_true', required=False, help="Add debugging log output.")
#     args = parser.parse_args()
    args = parser.parse_args(['--input', '/home/jfear/sandbox/secim/data/ST000015.tsv',
                              '--design', '/home/jfear/sandbox/secim/data/ST000015_design.tsv',
                              '--ID', 'Name',
                              '--out', '/home/jfear/sandbox/secim/data/test.csv',
                              '--debug'])
    return(args)


def main(args):
    """ """
    # Import data
    logger.info(u'Importing data with following parameters: \n\tWide: {0}\n\tDesign: {1}\n\tUnique ID: {2}'.format(args.fname, args.dname, args.uniqID))
    dat = wideToDesign(args.fname, args.dname, args.uniqID)

    # Only interested in samples
    wide = dat.wide[dat.sampleIDs]

    # Count the number of digits before decimal and get basic distribution info
    cnt = wide.applymap(lambda x: len(str(x).split('.')[0]))
    cnt['min'] = cnt.apply(np.min, axis=1)
    cnt['max'] = cnt.apply(np.max, axis=1)
    cnt['diff'] = cnt['max'] - cnt['min']

    # wite output
    cnt.to_csv(args.oname)


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
    else:
        sl.setLogger(logger)

    main(args)
