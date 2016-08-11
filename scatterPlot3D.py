#!/usr/bin/env python

# Built-in packages
from __future__ import division
import logging
import argparse
from argparse import RawDescriptionHelpFormatter
import os

# Add-on packages
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Local Packages
import logger as sl

def getOptions(myopts=None):
    """ Function to pull in arguments """
    description = """  """
    parser = argparse.ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)
    group1 = parser.add_argument_group(title='Standard input', description='Standard input for SECIM tools.')
    group1.add_argument("--input", dest="fname", action='store', required=True, help="Input dataset in wide format.")
    group1.add_argument("--ID", dest="uniqID", action='store', required=True, help="Name of the column with unique identifiers.")
    group1.add_argument("--design", dest="dname", action='store', required=False, help="Design file [Optional].")
    group1.add_argument("--group", dest="group", action='store', required=False, help="Group/treatment identifier in design file [Optional].")

    group2 = parser.add_argument_group(title='Required input', description='Additional required input for this tool.')
    group2.add_argument("--xaxis", dest="x", action='store', required=True, help="Column to use for x-axis.")
    group2.add_argument("--yaxis", dest="y", action='store', required=True, help="Column to use for y-axis.")
    group2.add_argument("--zaxis", dest="z", action='store', required=True, help="Column to use for z-axis.")
    group2.add_argument("--title", dest="title", action='store', required=False, help="Title for scatter plot [Optional].")
    group2.add_argument("--xname", dest="xlab", action='store', required=False, help="Name to use for x-axis [Optional].")
    group2.add_argument("--yname", dest="ylab", action='store', required=False, help="Name to use for y-axis [Optional].")
    group2.add_argument("--zname", dest="zlab", action='store', required=False, help="Name to use for z-axis [Optional].")

    group3 = parser.add_argument_group(title='Required output')
    group3.add_argument("--fig", dest="fig", action='store', required=True, help="Name of output figure.")

    group4 = parser.add_argument_group(title='Development Settings')
    group4.add_argument("--debug", dest="debug", action='store_true', required=False, help="Add debugging log output.")

    args = parser.parse_args()

    return args

def getColors(x):
    colors = ['b', 'r', 'c', 'm', 'y', 'k', 'b', 'g']
    cmap = dict()

    for group, color in zip(x, colors):
        cmap[group] = color

    return cmap


def galaxySavefig(fig, fname):
    """ Take galaxy DAT file and save as fig """

    png_out = fname + '.png'
    fig.savefig(png_out)

    # shuffle it back and clean up
    data = file(png_out, 'rb').read()
    with open(fname, 'wb') as fp:
        fp.write(data)
    os.remove(png_out)


def buildLegend(ax, cmap):
    plts = list()
    labels = list()
    for i, val in cmap.items():
        plts.append(matplotlib.lines.Line2D([0],[0], linestyle="none", c=val, marker = 'o'))
        labels.append(i)
    ax.legend(plts, labels, numpoints = 1)


def main(args):
    # Import data
    logger.info('Importing Data')
    dat = pd.read_table(args.fname, comment='#')
    dat.set_index(args.uniqID, inplace=True)

    # Prepare Figure
    ## Title
    if args.title:
        title = args.title
    else:
        title = '{0} vs {1} vs {2}'.format(args.x, args.y, args.z)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(title)
    if args.xlab:
        xlab = args.xlab
    else:
        xlab = args.x

    if args.ylab:
        ylab = args.ylab
    else:
        ylab = args.y

    if args.zlab:
        zlab = args.zlab
    else:
        zlab = args.z

    # Make plots
    if args.dname and args.group:
        # If group information give color by group.
        design = pd.read_table(args.dname)
        design.set_index('sampleID', inplace=True)
        merged = dat.join(design, how='left')
        grp = merged.groupby(args.group)
        cmap = getColors(grp.indices.keys())

        for i, val in grp:
            c = cmap[i]
            xs = val[args.x]
            ys = val[args.y]
            zs = val[args.z]
            ax.scatter(xs, ys, zs, c=c, s=100, label=i)
        buildLegend(ax, cmap)

    else:
        # Else just plot.
        xs = dat[args.x]
        ys = dat[args.y]
        zs = dat[args.z]
        ax.scatter(xs, ys, zs, s=100)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(ylab)

    galaxySavefig(fig, args.fig)


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
    else:
        sl.setLogger(logger)

    main(args)
