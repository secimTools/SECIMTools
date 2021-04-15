#!/usr/bin/env python
################################################################################
# AUTHORS:  Miguel Ibarra <miguelib@ufl.edu>
#           Matt Thoburn <mthoburn@ufl.edu>
#
# DESCRIPTION: This runs an Elastic Net or Lasso Test on wide data
################################################################################

import os
import logging
import argparse
try:
    from importlib import resources as ires
except ImportError:
    import importlib_resources as ires
import itertools as it
import numpy as np
import rpy2.robjects as robjects
from argparse import RawDescriptionHelpFormatter
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage as STAP
from rpy2.rinterface_lib.embedded import RRuntimeError
from secimtools.dataManager import logger as sl
from secimtools.dataManager.interface import wideToDesign


def getOptions(myOpts=None):
    description = """
    The tool performs feature selection using LASSO/Elastic Net feature selection method.
    """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=RawDescriptionHelpFormatter
    )
    # Standard Input
    standard = parser.add_argument_group(
        title="Standard input", description="Standard input for SECIM tools."
    )
    standard.add_argument(
        "-i",
        "--input",
        dest="input",
        action="store",
        required=True,
        help="Input dataset in wide format.",
    )
    standard.add_argument(
        "-d",
        "--design",
        dest="design",
        action="store",
        required=True,
        help="Design file.",
    )
    standard.add_argument(
        "-id",
        "--ID",
        dest="uniqID",
        action="store",
        required=True,
        help="Name of the column with unique" " identifiers.",
    )
    standard.add_argument(
        "-g",
        "--group",
        dest="group",
        action="store",
        required=False,
        default=False,
        help="Name of the column" " with groups.",
    )
    # Tool Input
    tool = parser.add_argument_group(title="Tool Especific")
    tool.add_argument(
        "-a",
        "--alpha",
        dest="alpha",
        type=float,
        action="store",
        required=True,
        help="Alpha Value.",
    )
    # Tool Output
    output = parser.add_argument_group(title="Required output")
    output.add_argument(
        "-c",
        "--coefficients",
        dest="coefficients",
        action="store",
        required=False,
        help="Path of en" " coefficients file.",
    )
    output.add_argument(
        "-f",
        "--flags",
        dest="flags",
        action="store",
        required=False,
        help="Path of en flag file.",
    )
    output.add_argument(
        "-p",
        "--plots",
        dest="plots",
        action="store",
        required=False,
        help="Path of en coefficients file.",
    )
    parser.add_argument(
        "-r",
        "--rscript",
        action="store",
        required=False,
        help="Full path to R script if not using package version",
    )

    args = parser.parse_args()

    # Standardize paths
    args.input = os.path.abspath(args.input)
    args.plots = os.path.abspath(args.plots)
    args.flags = os.path.abspath(args.flags)
    args.design = os.path.abspath(args.design)
    args.coefficients = os.path.abspath(args.coefficients)

    return args


def main(args):
    if not args.rscript:
        with ires.path("secimtools.data", "lasso_enet.R") as R_path:
            my_r_script_path = str(R_path)
    else:
        my_r_script_path = args.rscript
    logger.info(f"R script path: {my_r_script_path}")

    pandas2ri.activate()

    with open(my_r_script_path, "r") as f:
        rFile = f.read()
    lassoEnetScript = STAP(rFile, "lasso_enet")

    # Import data trough the interface module
    dat = wideToDesign(
        args.input, args.design, args.uniqID, group=args.group, logger=logger
    )

    dat.dropMissing()
    # Get remaining Sample IDs for dataframe filtering of irrelevant columns
    sample_ids = dat.wide.index.tolist()
    group_col_name = dat.group

    # Transpose Data so compounds are columns
    dat.trans = dat.transpose()
    group_data = dat.trans[group_col_name]
    dat.trans.columns.name = ""


    # Dropping nan columns from design
    removed = dat.design[dat.design[dat.group] == "nan"]
    dat.design = dat.design[dat.design[dat.group] != "nan"]
    dat.trans.drop(removed.index.values, axis=0, inplace=True)
    dat.trans = dat.trans.loc[:,sample_ids]
    dat.trans['group'] = group_data

    logger.info("{0} removed from analysis".format(removed.index.values))
    dat.design.rename(columns={dat.group: "group"}, inplace=True)
    dat.trans.rename(columns={dat.group: "group"}, inplace=True)

    groupList = [
        title for title, group in dat.design.groupby("group") if len(group.index) > 2
    ]

    # Turn the group list into pairwise combinations
    comboMatrix = np.array(list(it.combinations(groupList, 2)))
    comboLength = len(comboMatrix)

    correct_list_of_names = np.array(dat.trans.columns.values.tolist())
    try:
        returns = lassoEnetScript.lassoEN(
            dat.trans,
            dat.design,
            args.uniqID,
            correct_list_of_names,
            comboMatrix,
            comboLength,
            args.alpha,
            args.plots,
        )
    except RRuntimeError as e:
        try:
            e.context = {
                'r_traceback': '\n'.join((r'unlist(traceback())'))
            }
        except Exception as traceback_exc:
            e.context = {
                'r_traceback':
                    '(an error occurred while getting traceback from R)',
                'r_traceback_err': traceback_exc,
            }
        raise
    robjects.r["write.table"](
        returns[0],
        file=args.coefficients,
        sep="\t",
        quote=False,
        row_names=False,
        col_names=True,
    )
    robjects.r["write.table"](
        returns[1],
        file=args.flags,
        sep="\t",
        quote=False,
        row_names=False,
        col_names=True,
    )
    # Finishing
    logger.info("Script Complete!")


if __name__ == "__main__":
    args = getOptions()
    logger = logging.getLogger()
    sl.setLogger(logger)
    logger.info(
        "Importing data with the folowing parameters: "
        "\n\tWide:  {0}"
        "\n\tDesign:{1}"
        "\n\tUniqID:{2}"
        "\n\tAlpha: {3}".format(args.input, args.design, args.uniqID, args.alpha)
    )
    main(args)
