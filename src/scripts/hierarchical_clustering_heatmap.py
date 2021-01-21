#!/usr/bin/env python
################################################################################
#
# SCRIPT: hierarchical_clustering_heatmap.py
#
# AUTHORS: Miguel A Ibarra <miguelib@ufl.edu>
#          Oleksandr Moskalenko <om@rc.ufl.edu>
#
# DESCRIPTION: This script takes a a wide format file and plots a heatmap of
#               the data. This tool is usefull to get a representation of the
#               data.
#
#     input     : Path to Wide formated file.
#     design    : Path to Design file corresponding to wide file.
#     id        : Uniq ID on wide file (ie. RowID).
#     heatmap   : Output path for loads heatmap.
#
# The output is a heatmap that describes the wide dataset.
################################################################################

import os
import argparse
import logging
import matplotlib

matplotlib.use("Agg")
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from secimtools.dataManager import logger as sl
from secimtools.dataManager.interface import wideToDesign
from secimtools.visualManager import module_heatmap as hm
from secimtools.visualManager.manager_color import colorHandler
from secimtools.visualManager.manager_figure import figureHandler


def getOptions():
    """Function to pull arguments"""
    parser = argparse.ArgumentParser(
        description="""
    Takes a wide and design file and plots a heatmap of the data """
    )
    # Standar Input
    standar = parser.add_argument_group(
        title="Standard input", description="Standard input for SECIM tools."
    )
    standar.add_argument(
        "-i",
        "--input",
        dest="input",
        action="store",
        required=True,
        help="Input dataset in wide format.",
    )
    standar.add_argument(
        "-d",
        "--design",
        dest="design",
        action="store",
        required=True,
        help="Design file.",
    )
    standar.add_argument(
        "-id",
        "--uniqID",
        dest="uniqID",
        action="store",
        required=True,
        help="Name of the column with unique " "dentifiers.",
    )
    # Tool Input
    tool = parser.add_argument_group(
        title="Tool input", description="Input specific for this tool"
    )
    tool.add_argument(
        "-den",
        "--dendogram",
        dest="dendogram",
        action="store_true",
        required=False,
        help="Indicate wether you want to use " "a dendogram or not in the heatmap.",
    )
    tool.add_argument(
        "-l",
        "--labels",
        dest="labels",
        action="store",
        default="None",
        required=False,
        choices=["x", "y", "x,y", "None"],
        help="Indicate wich" "labels if any that you want to remove from the plot.",
    )
    # Tool Output
    output = parser.add_argument_group(
        title="Output paths", description="Paths for the output files"
    )
    output.add_argument(
        "-f",
        "--fig",
        dest="fig",
        action="store",
        required=True,
        help="Output path for heatmap file[PDF]",
    )
    # Plot Options
    plot = parser.add_argument_group(title="Plot options")
    plot.add_argument(
        "-pal",
        "--palette",
        dest="palette",
        action="store",
        required=False,
        default="diverging",
        help="Name of the palette to use.",
    )
    plot.add_argument(
        "-col",
        "--color",
        dest="color",
        action="store",
        required=False,
        default="Spectral_10",
        help="Name of a valid color scheme" " on the selected palette",
    )
    args = parser.parse_args()

    # Standardize paths
    args.fig = os.path.abspath(args.fig)
    args.input = os.path.abspath(args.input)
    args.design = os.path.abspath(args.design)

    return args


def main(args):
    """Runs eveything"""
    # Importing data
    dat = wideToDesign(args.input, args.design, args.uniqID, logger=logger)

    # Cleaning from missing data
    dat.dropMissing()

    # Getting labels to drop from arguments
    x = True
    y = True
    if "x" in args.labels:
        x = False
    if "y" in args.labels:
        y = False

    # Plotting with dendogram Hierarchical cluster heatmap (HCH)
    logger.info("Plotting heatmaps")
    if args.dendogram == True:
        fh = hm.plotHCHeatmap(
            dat.wide, hcheatmap=True, cmap=palette.mpl_colormap, xlbls=x, ylbls=y
        )
        fh.savefig(args.fig, format="pdf")

    # Plotting without a dendogram single heatmap
    else:
        fh = figureHandler(proj="2d", figsize=(14, 14))
        hm.plotHeatmap(dat.wide, fh.ax[0], cmap=palette.mpl_colormap, xlbls=x, ylbls=y)
        fh.formatAxis(xTitle="sampleID")
        fh.export(out=args.fig, dpi=300)
    logger.info("Script Complete!")


if __name__ == "__main__":
    args = getOptions()
    logger = logging.getLogger()
    sl.setLogger(logger)
    logger.info(
        "Importing data with following parameters: "
        "\n\tInput: {0}"
        "\n\tDesign: {1}"
        "\n\tuniqID: {2}".format(args.input, args.design, args.uniqID)
    )
    palette = colorHandler(pal=args.palette, col=args.color)
    logger.info(
        "Using {0} color scheme from {1} palette".format(args.color, args.palette)
    )
    main(args)
