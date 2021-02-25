#!/usr/bin/env python
####################################################################
# AUTHOR: Miguel Ibarra (miguelib@ufl.edu)
#
# DESCRIPTION: Pairwise and to mean standarized euclidean comparison
####################################################################

import os
import logging
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy.linalg import svd
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.neighbors import DistanceMetric
from secimtools.dataManager import logger as sl
from secimtools.dataManager.interface import wideToDesign
from secimtools.visualManager import module_box as box
from secimtools.visualManager import module_lines as lines
from secimtools.visualManager import module_scatter as scatter
from secimtools.visualManager.manager_color import colorHandler
from secimtools.visualManager.manager_figure import figureHandler


def getOptions():
    """ Function to pull in arguments """
    description = """"""

    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Standard Input
    standard = parser.add_argument_group(description="Required Input")
    standard.add_argument(
        "-i",
        "--input",
        dest="input",
        action="store",
        required=True,
        help="Dataset in Wide format",
    )
    standard.add_argument(
        "-d",
        "--design", dest="design",
        action="store",
        required=True,
        help="Design file",
    )
    standard.add_argument(
        "-id",
        "--ID",
        dest="uniqID",
        action="store",
        required=True,
        help="Name of the column with uniqueID.",
    )
    standard.add_argument(
        "-g",
        "--group",
        dest="group",
        default=False,
        action="store",
        required=False,
        help="Treatment group",
    )
    standard.add_argument(
        "-o", "--order", dest="order", action="store", default=False, help="Run Order"
    )
    standard.add_argument(
        "-l",
        "--levels",
        dest="levels",
        action="store",
        default=False,
        help="Additional notes.",
    )
    # Tool output
    output = parser.add_argument_group(description="Output Files")
    output.add_argument(
        "-f",
        "--figure",
        dest="figure",
        action="store",
        required=True,
        help="PDF Output of standardized" "Euclidean distance plot",
    )
    output.add_argument(
        "-m",
        "--distanceToMean",
        dest="toMean",
        action="store",
        required=True,
        help="TSV Output of Mahalanobis " "distances from samples to the mean.",
    )
    output.add_argument(
        "-pw",
        "--distancePairwise",
        dest="pairwise",
        action="store",
        required=True,
        help="TSV Output of sample-pairwise" "mahalanobis distances.",
    )
    # Tool Input
    tool = parser.add_argument_group(description="Optional Input")
    tool.add_argument(
        "-p",
        "--per",
        dest="p",
        action="store",
        required=False,
        default=0.95,
        type=float,
        help="The threshold" "for standard distributions. The default is 0.95.",
    )
    tool.add_argument(
        "-pen",
        "--penalty",
        dest="penalty",
        action="store",
        required=False,
        default=0.5,
        type=float,
        help="Value" " of lambda for the penalty.",
    )
    tool.add_argument(
        "-lg",
        "--log",
        dest="log",
        action="store",
        required=False,
        default=True,
        help="Log file",
    )
    # Plot options
    plot = parser.add_argument_group(title="Plot options")
    plot.add_argument(
        "-pal",
        "--palette",
        dest="palette",
        action="store",
        required=False,
        default="tableau",
        help="Name of the palette to use.",
    )
    plot.add_argument(
        "-col",
        "--color",
        dest="color",
        action="store",
        required=False,
        default="Tableau_20",
        help="Name of a valid color scheme" " on the selected palette",
    )
    args = parser.parse_args()

    # Standardize paths
    args.figure = os.path.abspath(args.figure)
    args.toMean = os.path.abspath(args.toMean)
    args.pairwise = os.path.abspath(args.pairwise)

    # if args.levels then split otherwise args.level = emptylist
    if args.levels:
        args.levels = args.levels.split(",")

    return args


def calculatePenalizedSigma(data, penalty=0.5):
    # Getting n and p of data.
    n, p = data.shape

    # Calculate the mean of every row
    data["mean"] = data.mean(axis=1)

    # Standardize data (_std stands for standardized)
    data_std = data.apply(lambda x: x - x["mean"], axis=1)

    # Dropping mean column from both data and data_standardized
    data.drop("mean", axis=1, inplace=True)
    data_std.drop("mean", axis=1, inplace=True)

    # Calculate singular value decomposition
    U, s, V = svd(data_std)

    # Calculate ds based on ss (d = s**2)
    d = s ** 2

    # Based on ds calculate the penalty. penalty must be expressed as a
    # proportion (from 0 to 1) to use it on the np.percentile it will be
    # multiplied by 100
    penalty = np.percentile(d, q=penalty * 100.0)

    # After the calculation of the penalty extend d vector to size n
    d = np.append(d, np.zeros(shape=(n - len(d))))

    # Calculate penalty and inverse for d (1/(d+penalty))
    d = [1 / (val + penalty) for val in d]

    # Creating empty matrix of size n x p and the fiiling the diagonal with
    # penalized s values
    # S = np.zeros((n,p))
    # S[:len(s),:len(s)] = np.diag(s)
    D = np.diag(d)

    # Compute the stimate of the penalized sigma
    penalized_sigma = np.dot(U, np.dot(D, U.T))

    # Multiply everything by p-1
    penalized_sigma = (p - 1) * penalized_sigma

    # Returning penalized sigma
    return penalized_sigma


def calculateDistances(data, V_VI):
    """
    Calculates euclidean or mahalanobis distances. Returns an array of
    distances to the Mean and an a matrix of pairwise distances.

    :Arguments:
        :type wide: pandas.DataFrame
        :param wide: A wide formatted data frame with samples as columns and
                     compounds as rows.

    :Returns:
        :return distanceToMean: pd.DataFrames with distances to the mean.
        :rtype: pd.DataFrames

        :return distancePairwise: pd.DataFrames with pairwisde distances between
        samples.
        :rtype: pd.DataFrames
    """
    # Calculating mean
    mean = pd.DataFrame(data.mean(axis=1))

    # Getting metric
    dist = DistanceMetric.get_metric("mahalanobis", VI=V_VI)

    # Calculate distance from all samples to the mean
    distanceToMean = dist.pairwise(data.values.T, mean.T)
    distanceToMean = pd.DataFrame(
        distanceToMean, columns=["distance_to_mean"], index=data.columns
    )
    distanceToMean.name = data.name

    # Calculate pairwise distances among samples
    distancePairwise = dist.pairwise(data.values.T)
    distancePairwise = pd.DataFrame(
        distancePairwise, columns=data.columns, index=data.columns
    )
    distancePairwise.name = data.name

    # Converts to NaN the diagonal
    for index, row in distancePairwise.iterrows():
        distancePairwise.loc[index, index] = np.nan

    return (distanceToMean, distancePairwise)


def calculateCutoffs(data, p):
    """
    Calculate the Standardized Euclidean Distance and return an array of
    distances to the Mean and a matrix of pairwise distances.

    :Arguments:
        :type wide: pandas.DataFrame
        :param wide: A wide formatted data frame with samples as columns and
                     compounds as rows.

        :type p: float.
        :param p: percentile of cutoff.

    :Returns:
        :rtype cutoff1: pandas.dataFrame
        :return cutoff1: Cutoff values for mean, beta, chi-sqr and normal.

        :rtype cutoff2: pandas.dataFrame
        :return cutoff2: Cutoff values for pairwise, beta, chi-sqr and normal.
    """

    # Stablish iterations, and numer of colums ps and number of rows nf
    ps = len(data.columns)  # p  number of samples
    nf = len(data.index)  # s number of features
    iters = 20000

    # Calculates betaP
    betaP = np.percentile(
        pd.DataFrame(
            stats.beta.rvs(0.5, 0.5 * (ps - 2), size=iters * nf).reshape(iters, nf)
        ).sum(axis=1),
        p * 100.0,
    )

    # casting to float so it behaves well
    ps = float(ps)
    nf = float(nf)

    # Calculates cutoffs beta,norm & chisq for data to mean
    betaCut1 = np.sqrt((ps - 1) ** 2 / ps * betaP)
    normCut1 = np.sqrt(
        stats.norm.ppf(
            p,
            (ps - 1) / ps * nf,
            np.sqrt(2 * nf * (ps - 2) * (ps - 1) ** 2 / ps ** 2 / (ps + 1)),
        )
    )
    chisqCut1 = np.sqrt((ps - 1) / ps * stats.chi2.ppf(p, nf))

    # Calculates cutoffs beta,n norm & chisq for pairwise
    betaCut2 = np.sqrt((ps - 1) * 2 * betaP)
    normCut2 = np.sqrt(stats.norm.ppf(p, 2 * nf, np.sqrt(8 * nf * (ps - 2) / (ps + 1))))
    chisqCut2 = np.sqrt(2 * stats.chi2.ppf(p, nf))

    # Create data fram for ecah set of cut offs
    cutoff1 = pd.DataFrame(
        [[betaCut1, normCut1, chisqCut1], ["Beta(Exact)", "Normal", "Chi-sq"]],
        index=["cut", "name"],
        columns=["Beta(Exact)", "Normal", "Chi-sq"],
    )
    cutoff2 = pd.DataFrame(
        [[betaCut2, normCut2, chisqCut2], ["Beta(Exact)", "Normal", "Chi-sq"]],
        index=["cut", "name"],
        columns=["Beta(Exact)", "Normal", "Chi-sq"],
    )

    # Create Palette
    cutPalette.getColors(cutoff1.T, ["name"])

    # Returning colors
    return (cutoff1, cutoff2)


def plotCutoffs(cut_S, ax, p):
    """
    Plot the cutoff lines to each plot

    :Arguments:
        :type cut_S: pandas.Series
        :param cut_S: contains a cutoff value, name and color

        :type ax: matplotlib.axes._subplots.AxesSubplot
        :param ax: Gets an ax project.

        :type p: float
        :param p: percentile of cutoff
    """
    lines.drawCutoffHoriz(
        ax=ax,
        y=float(cut_S.values[0]),
        cl=cutPalette.ugColors[cut_S.name],
        lb="{0} {1}% Threshold: {2}".format(
            cut_S.name, round(p * 100, 3), round(float(cut_S.values[0]), 1)
        ),
        ls="--",
        lw=2,
    )


def plotDistances(df_distance, palette, plotType, disType, cutoff, p, pdf):
    # Geting number of samples in dataframe (ns stands for number of samples)
    ns = len(df_distance.index)

    # Calculates the widht for the figure base on the number of samples
    figWidth = max(ns / 2, 16)

    # Keeping the order on the colors
    df_distance["colors"] = palette.design["colors"]

    # Create figure object with a single axis
    figure = figureHandler(proj="2d", figsize=(figWidth, 8))

    # Getting type of distance file
    if "distance_to_mean" in df_distance.columns:
        dataType = "to the mean"
    else:
        dataType = "pairwise"

    # Getting ty of distance header
    if disType == "Mahalanobis":
        distType1 = "Penalized"
        distType2 = disType
    else:
        distType1 = "Standardized"
        distType2 = disType

    # Adds Figure title, x axis limits and set the xticks
    figure.formatAxis(
        figTitle="{0} for {1} {2} Distance for {3} {4}".format(
            plotType, distType1, distType2, df_distance.name, dataType
        ),
        yTitle="{0} {1} Distance".format(distType1, distType2),
        xTitle="Index",
        ylim="ignore",
        xlim=(-0.5, -0.5 + ns),
        xticks=df_distance.index,
    )

    # If distance to mean
    if dataType == "to the mean":
        # Plot scatterplot quickplot
        scatter.scatter2D(
            ax=figure.ax[0],
            colorList=df_distance["colors"],
            x=range(len(df_distance.index)),
            y=df_distance["distance_to_mean"],
        )
    # if pairwise
    else:
        if plotType == "Scatterplot":
            # Plot scatterplot
            for index in df_distance.index:
                scatter.scatter2D(
                    ax=figure.ax[0],
                    colorList=df_distance["colors"][index],
                    x=range(len(df_distance.index)),
                    y=df_distance[index],
                )

        elif plotType == "Box-plots":
            # Plot Box plot
            box.boxDF(ax=figure.ax[0], colors=df_distance["colors"], dat=df_distance)

    # Shrink figure
    figure.shrink()

    # Plot legend
    figure.makeLegend(figure.ax[0], palette.ugColors, palette.combName)

    # Add a cutoof line
    cutoff.apply(lambda x: plotCutoffs(x, ax=figure.ax[0], p=p), axis=0)

    # Add figure to PDF and close the figure afterwards
    figure.addToPdf(pdf)

    # Drop "color" column to no mess the results
    df_distance.drop("colors", axis=1, inplace=True)


def main(args):
    """
    Main function
    """
    if args.levels and args.group:
        levels = [args.group] + args.levels
    elif args.group and not args.levels:
        levels = [args.group]
    else:
        levels = []
    logger.info(u"Color selection groups: {0}".format(",".join(levels)))

    dat = wideToDesign(
        args.input,
        args.design,
        args.uniqID,
        group=args.group,
        anno=args.levels,
        logger=logger,
        runOrder=args.order,
    )

    # Removing groups with just one sample and then clean from missing data.
    dat.removeSingle()
    dat.dropMissing()

    # Select colors for data by adding an additional column for colors
    dataPalette.getColors(design=dat.design, groups=levels)

    if args.group:
        disGroups = [
            (group.index, level)
            for level, group in dataPalette.design.groupby(dataPalette.combName)
        ]
    else:
        disGroups = [(dat.design.index, "samples")]

    pairwise_disCuts = list()
    toMean_disCuts = list()
    for indexes, name in disGroups:
        # If less than 3 elements in the group skip to the next
        if len(indexes) < 3:
            logger.error(
                "Group {0} with fewer than 3 elements is excluded from the analysis".format(name)
            )
            continue

        # Subsetting wide
        currentFrame = pd.DataFrame(dat.wide[indexes].copy())
        currentFrame.name = name

        # Calculate Penalized Sigma
        penalizedSigma = calculatePenalizedSigma(
            data=currentFrame, penalty=args.penalty
        )

        # Calculate Distances (dis stands for distance)
        disToMean, disPairwise = calculateDistances(
            data=currentFrame, V_VI=penalizedSigma
        )

        # Calculate cutoffs
        cutoff1, cutoff2 = calculateCutoffs(currentFrame, args.p)

        # Appending results
        pairwise_disCuts.append([disPairwise, cutoff2])
        toMean_disCuts.append([disToMean, cutoff1])

    if args.group:
        # Splitting results to mean and pairwise
        pairwise_dis = [distance for distance, cutoff in pairwise_disCuts]
        toMean_dis = [distance for distance, cutoff in toMean_disCuts]

        # Merging to get distance for all pairwise
        pairwise_dis_all = pd.DataFrame()
        for dis in pairwise_dis:
            pairwise_dis_all = pd.DataFrame.merge(
                pairwise_dis_all,
                dis,
                left_index=True,
                right_index=True,
                how="outer",
                sort=False,
            )
        pairwise_dis_all.name = "samples"

        # Merging to get distance for all to mean
        toMean_dis_all = pd.DataFrame(columns=["distance_to_mean","group"])
        for dis in toMean_dis:
            dis["group"] = dis.name
            toMean_dis_all = toMean_dis_all.append(dis)
        toMean_dis_all.sort_values(by="group", inplace=True)
        toMean_dis_all.drop("group", axis=1, inplace=True)
        toMean_dis_all.name = "samples"

        # Get cuttoffs for distances
        cutoff1, cutoff2 = calculateCutoffs(dat.wide, args.p)

        # Append toMean_dis_all and pairwise_dis_all to toMean_dis_cuts and
        # pairwise_dis_cuts respectively.
        toMean_disCuts.append([toMean_dis_all, cutoff1])
        pairwise_disCuts.append([pairwise_dis_all, cutoff2])

    # Iterate over each pair of (distance,cutoff) for toMean and pairwise to
    # plot  distances.
    with PdfPages((args.figure)) as pdf:
        # Iterating over toMean,pairwise distances in parallel
        for toMean, pairwise in zip(toMean_disCuts, pairwise_disCuts):
            # Making plots
            plotDistances(
                df_distance=toMean[0],
                palette=dataPalette,
                p=args.p,
                plotType="Scatterplot",
                disType="Mahalanobis",
                cutoff=toMean[1],
                pdf=pdf,
            )
            plotDistances(
                df_distance=pairwise[0],
                palette=dataPalette,
                p=args.p,
                plotType="Scatterplot",
                disType="Mahalanobis",
                cutoff=pairwise[1],
                pdf=pdf,
            )
            plotDistances(
                df_distance=pairwise[0],
                palette=dataPalette,
                p=args.p,
                plotType="Box-plots",
                disType="Mahalanobis",
                cutoff=pairwise[1],
                pdf=pdf,
            )

    # Since its a list of dataframes and we are only interested in the last one
    # we are using [-1] to access it and [0] to getit out of the list.
    # Outputting distances to mean and pairwise
    toMean_disCuts[-1][0].to_csv(args.toMean, index_label="sampleID", sep="\t")
    pairwise_disCuts[-1][0].to_csv(args.pairwise, index_label="sampleID", sep="\t")

    # Ending script
    logger.info("Script complete.")


if __name__ == "__main__":
    args = getOptions()
    logger = logging.getLogger()
    sl.setLogger(logger)
    logger.info(
        u"""Importing data with following parameters:
                \tWide: {0}
                \tDesign: {1}
                \tUnique ID: {2}
                \tGroup: {3}
                \tRun Order: {4}
                """.format(
            args.input, args.design, args.uniqID, args.group, args.order
        )
    )
    dataPalette = colorHandler(pal=args.palette, col=args.color)
    cutPalette = colorHandler(pal="tableau", col="TrafficLight_9")
    logger.info(
        u"Using {0} color scheme from {1} palette".format(args.color, args.palette)
    )
    main(args)
