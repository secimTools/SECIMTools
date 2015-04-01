#!/usr/bin/env python

# Built-in packages
import logging
import argparse
from argparse import RawDescriptionHelpFormatter

# Add-on packages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpld3

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
    parser.add_argument("--fig", dest="ofig", action='store', required=True, help="Output figure name [pdf].")
    parser.add_argument("--fig2", dest="ofig2", action='store', required=True, help="Output figure name [html].")
    parser.add_argument("--debug", dest="debug", action='store_true', required=False, help="Add debugging log output.")
    args = parser.parse_args()
#     args = parser.parse_args(['--input', '/home/jfear/sandbox/secim/data/ST000015_AN000032_v2.tsv',
#                               '--design', '/home/jfear/sandbox/secim/data/ST000015_design_v2.tsv',
#                               '--ID', 'Name',
#                               '--group', 'treatment',
#                               '--fig', '/home/jfear/sandbox/secim/data/test.pdf',
#                               '--fig2', '/home/jfear/sandbox/secim/data/test2.html',
#                               '--debug'])
    return(args)


def main(args):
    # Import data
    logger.info(u'Importing data with following parameters: \n\tWide: {0}\n\tDesign: {1}\n\tUnique ID: {2}\n\tGroup Column: {3}'.format(args.fname, args.dname, args.uniqID, args.group))
    dat = wideToDesign(args.fname, args.dname, args.uniqID, args.group, clean_string=True)
    dat.wide.convert_objects(convert_numeric=True)

    samples = dat.wide[dat.sampleIDs]
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    samples.plot(kind='kde', title='Distribution by Sample', ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=5, loc='upper right', fontsize=10)

    plt.savefig(args.ofig, format='pdf')
    mpld3.save_html(fig, args.ofig2)


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
    else:
        sl.setLogger(logger)

    main(args)
