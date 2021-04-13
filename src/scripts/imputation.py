#!/usr/bin/env python
######################################################################################
#
# MODULE: imputation.py
#
# AUTHORS: Matt Thoburn <mthoburn@ufl.edu>
#          Miguel Ibarra <miguelib@ufl.edu>
#
# DESCRIPTION: This attempts to impute missing data by an algorithm of the user's choice
#######################################################################################


import os
import sys
import logging
import argparse
from argparse import RawDescriptionHelpFormatter
import numpy as np
import pandas as pd
import pymc
from pymc import MCMC
from pymc.distributions import Impute
from pymc import Poisson, Normal, DiscreteUniform
import rpy2
import rpy2.robjects.numpy2ri
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
from secimtools.dataManager import logger as sl
from secimtools.dataManager.interface import wideToDesign


def getOptions(myOpts=None):
    description = """
    The tool performs imputations using selected algorith
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
        help="Name of the column with unique identifiers.",
    )
    standard.add_argument(
        "-g",
        "--group",
        dest="group",
        action="store",
        required=False,
        default=False,
        help="Name of the column with groups.",
    )
    # Tool Output
    output = parser.add_argument_group(title="Required output")
    output.add_argument(
        "-o",
        "--output",
        dest="output",
        action="store",
        required=False,
        help="Path of output file.",
    )
    # Tool Input
    tool = parser.add_argument_group(
        title="Tool input", description="Tool specific input."
    )
    tool.add_argument(
        "-s",
        "--strategy",
        dest="strategy",
        action="store",
        required=True,
        choices=["knn", "mean", "median", "bayesian"],
        default=None,
        help="Imputation strategy: KNN, mean, " "median, or most frequent",
    )
    tool.add_argument(
        "-noz",
        "--no_zero",
        dest="noZero",
        action="store_true",
        required=False,
        default=True,
        help="Treat 0 as missing?",
    )
    tool.add_argument(
        "-noneg",
        "--no_negative",
        dest="noNegative",
        action="store_true",
        required=False,
        default=True,
        help="Treat negative as missing?",
    )
    tool.add_argument(
        "-ex",
        "--exclude",
        dest="exclude",
        action="store",
        required=False,
        default=False,
        help="Additional values to treat as missing" "data, seperated by commas",
    )
    tool.add_argument(
        "-rc",
        "--row_cutoff",
        dest="rowCutoff",
        action="store",
        required=False,
        default=0.5,
        type=float,
        help="Percent cutoff for "
        "imputation of rows.If this is exceeded, imputation will"
        "be done by mean instead of knn. Default: .5",
    )
    tool.add_argument(
        "-dist",
        "--distribution",
        dest="dist",
        required=False,
        default="poisson",
        choices=["Poisson", "Normal"],
        help="use mean or median to generate mu value for " "bayesian imputation",
    )
    # KNN Input
    knn = parser.add_argument_group(title="KNN input")
    knn.add_argument(
        "-k",
        "--knn",
        dest="knn",
        action="store",
        required=False,
        default=5,
        help="Number of nearest neighbors to search Default: 5.",
    )
    knn.add_argument(
        "-cc",
        "--col_cutoff",
        dest="colCutoff",
        action="store",
        required=False,
        default=0.8,
        help="Percent cutoff for"
        "imputation of columns. If this is exceeded, imputation"
        "will be done by mean instead of knn. Default: .8",
    )
    args = parser.parse_args()

    # Standardize paths
    args.input = os.path.abspath(args.input)
    args.design = os.path.abspath(args.design)
    args.output = os.path.abspath(args.output)

    return args


def preprocess(noz, non, ex, data):
    """
    Preprocesses data to replace all unaccepted values with np.nan so they can be imputedDataAsNumpy

    :Arguments:
        :type noz: bool
        :param noz: remove zeros?

        :type non: bool
        :param non: remove negative numbers?

        :type ex: string
        :param ex: custom characters to be removed. Will be parsed to a list of strings

        :type data: pandas DataFrame
        :param data: data to be imputed

    :Return:
        :type data: pandas DataFrame
        :param data: data to be imputed
    """

    # The rest of the values will be converted to float
    data = data.applymap(float)

    # All string instances on data will be set to nans
    data = data.applymap(lambda x: np.nan if isinstance(x, str) else x)

    # If non zero then convert al 0's to nans
    if noz:
        data = data.where(data != 0, np.nan)

    # if non negative numbers then convert al negative numbers to nans
    if non:
        data = data.where(data > 0, np.nan)

    # If a custom character is to be removed
    if ex:
        exclude = ex.split(",")
        data = data.applymap(lambda x: np.nan if x in exclude else x)
    return data


def imputeKNN(rc, cc, k, dat):
    """
    Imputes by K-Nearest Neighbors algorithm

    :Arguments:
        :type rc: float
        :param rc: row cutoff value that determines whether or not to default to mean imputation

        :type cc: float
        :param cc: column cutoff value that determines wheter or not to default to mean imputation

        :type k: int
        :param k: Number of nearby neighbors to consider when performing imputation

        :type dat: interface wideToDesign file
        :param dat: wide and design data bundled together

    :Returns:
        :type pdFull: pandas DataFrame
        :param pdFull: data with missing values imputed
    """
    # Configuring Rpy2
    logger.info("Configuring R")
    rpy2.robjects.numpy2ri.activate()

    # Import R packages
    base = importr("base")
    utils = importr("utils")
    robjects.r("library(impute)")

    # Creating a list with all the different groups
    # once inputed they will be concatenated back
    fixedFullDataset = list()

    out = sys.stdout  # Save the stdout path for later, we're going to need it
    f = open("/dev/null", "w")  # were going to use this to redirect stdout temporarily

    logger.info("Running KNN imputation")
    # Iterating over groups
    for title, group in dat.design.groupby(dat.group):
        # If len of the group then do not inpute
        if len(group.index) == 1:  # No nearby neighbors to impute
            logger.info(title + " has no neighbors, will not impute")
            fixedFullDataset.append(dat.wide[group.index])
            continue

        # If group len is not enough for k then use len - 1
        if len(group.index) <= k:  # some nearby, but not enough to use user specified k
            logger.info(
                title + " group length less than k, will use group length - 1 instead"
            )
            k = len(group.index) - 1

        # Convert wide data to a matrix
        wideData = dat.wide[group.index].values

        # Getting number of rows and columns in wide matrix
        numRows, numCols = wideData.shape

        # Creatting R objects
        matrixInR = robjects.r["matrix"](wideData, nrow=numRows, ncol=numCols)
        imputeKNN = robjects.r("impute.knn")

        # Impute on R module
        sys.stdout = f
        imputedObject = imputeKNN(data=matrixInR, k=k, rowmax=rc, colmax=cc)
        sys.stdout = out

        # Taking the inputed object back to python
        imputedDataAsNumpy = np.array(imputedObject[0])

        # Saving the inputed data as pandas DataFrame
        imputedDataAsPandas = pd.DataFrame(
            imputedDataAsNumpy,
            index=dat.wide[group.index].index,
            columns=dat.wide[group.index].columns,
        )

        # Apending the inputed data to the full pandas datset
        fixedFullDataset.append(imputedDataAsPandas)

        # reset k back to normal if it was modified @125
        k = int(args.knn)

        # Concatenating list of results to full dataframe again
        pdFull = pd.concat(fixedFullDataset, axis=1)

    # Returning pandas dataframe
    return pdFull


def imputeRow(row, rc, strategy, dist=False):
    """
        :Arguments:
        :type row: pandas.Series
        :param row: row to be imputed

        :type rc: float
        :param rc: row cutoff value that determines whether or not to default to
                     mean imputation

        :type strategy: str
        :param strategy: Strategy to be used for imputation.

        :type dist: str
        :param dist: Type of distribution to be used in Bayesian imbutation.

    :Returns:
        :type pdFull: pandas.Series
        :param pdFull: Imputed row.

    """
    # If ratio of missing/present values is greater than the row cutoff then
    # don't impute and save the row as it is
    if float(row.isnull().sum()) / float(len(row)) > rc:
        return row

    # If ratio is less than the rc then review for impute
    else:
        # If there's any missing value impute with either mean, median or bayesian.
        if row.isnull().any():
            if strategy == "mean":
                row.fillna(np.nanmean(row), inplace=True)
            elif strategy == "median":
                row.fillna(np.nanmedian(row), inplace=True)
            elif strategy == "bayesian":
                row = imputeBayesian(row=row, dist=dist)
            return row
        # if the row is complete the return the row as it is
        else:
            return row


def imputeBayesian(row, dist):
    out = sys.stdout  # Save the stdout path for later, we're going to need it
    f = open("/dev/null", "w")  # were going to use this to redirect stdout

    # filling nan with 0 so everything works
    row.fillna(0, inplace=True)

    # Masked Values
    maskedValues = np.ma.masked_equal(row.values, value=0)

    # Choose between distributions, either normal or Poisson.
    if dist == "Normal":

        # Calculate tau
        if np.std(maskedValues) == 0:
            tau = np.square(1 / (np.mean(maskedValues) / 3))
        else:
            tau = np.square((1 / (np.std(maskedValues))))

        # Uses only mean
        x = Impute("x", Normal, maskedValues, tau=tau, mu=np.mean(maskedValues))

    # For Poisson
    elif dist == "Poisson":
        x = Impute("x", Poisson, maskedValues, mu=np.mean(maskedValues))

    # Fancy test
    sys.stdout = f  # Skipin stdout
    m = MCMC(x)
    m.sample(iter=1, burn=0, thin=1)
    sys.stdout = out  # coming back

    # Getting list of missing values
    missing = [i for i in range(len(row.values)) if row.values[i] == 0]

    # Getting the imputed values from the model
    for i in range(len(missing)):
        keyString = "x[" + str(missing[i]) + "]"
        imputedValue = m.trace(keyString)[:]
        row.iloc[missing[i]] = imputedValue[0]

    # Returning to use nans
    row.replace(0, np.nan, inplace=True)
    return row


def iterateGroups(dat, strategy, rc, dist=False):
    # Create a list to concatenate all the results
    imputed = list()

    # Iterating over groups
    for title, group in dat.design.groupby(dat.group):
        # Getting current group
        currentGroup = dat.wide[group.index]

        # Try to impute only if the amount of columns in the group > 1
        if len(currentGroup.index) > 1:

            # Doing  mean/median imputation
            currentGroup.apply(imputeRow, args=(rc, strategy, dist), axis=1)

        # Appending imputed group to list of groups
        imputed.append(currentGroup)

    # Concatenate results into one single dataframe that contains all the
    # Columns of the orignal one but with imputed values when possible
    imputed_df = pd.concat(imputed, axis=1)

    # Returning full data frame
    return imputed_df


def main(args):
    # Import data with interface
    logger.info("Importig data with interface")
    dat = wideToDesign(
        args.input, args.design, uniqID=args.uniqID, group=args.group, logger=logger
    )

    # Preprocessing
    logger.info("Preprocessing")
    dat.wide = preprocess(
        noz=args.noZero, non=args.noNegative, ex=args.exclude, data=dat.wide
    )

    # Choosing knn as imputation method
    logger.info("Inpute")
    if args.strategy == "knn":
        pdFull = imputeKNN(
            rc=float(args.rowCutoff), cc=float(args.colCutoff), k=int(args.knn), dat=dat
        )
    else:
        # Iterate over groups and perform either a mean or median imputation.
        pdFull = iterateGroups(
            dat=dat, strategy=args.strategy, dist=args.dist, rc=args.rowCutoff
        )

    # Convert dataframe to float and round results to 4 digits
    pdFull.applymap(float)
    pdFull = pdFull.round(4)

    # Make sure that the output has the same unique.ID
    pdFull.index.name = args.uniqID

    # Saving inputed data
    pdFull.to_csv(args.output, sep="\t")
    logger.info("Script Complete!")


if __name__ == "__main__":
    args = getOptions()
    logger = logging.getLogger()
    sl.setLogger(logger)
    logger.info(
        "Importing data with following parameters:"
        "\n\tInput: {0}"
        "\n\tDesign: {1}"
        "\n\tuniqID: {2}"
        "\n\tgroup: {3}".format(args.input, args.design, args.uniqID, args.group)
    )
    main(args)
