#!/usr/bin/env python
######################################################################################
#
# SCRIPT: kruscal_wallis.py
#
# AUTHOR: Alexander Kirpich <akirpich@ufl.edu>
#
# DESCRIPTION: This tool runs a Kruscal-Wallis test on the data for each feature (row).
#              The test is nonparametric analog to One Way ANOVA model.
#
#######################################################################################

import os
import logging
import argparse
import warnings
from itertools import combinations
from argparse import RawDescriptionHelpFormatter
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats.mstats import kruskalwallis
from secimtools.visualManager import module_lines as lines
from secimtools.visualManager import module_scatter as scatter
from secimtools.visualManager.manager_color import colorHandler
from secimtools.visualManager.manager_figure import figureHandler
from secimtools.dataManager import logger as sl
from secimtools.dataManager.interface import wideToDesign


def getOptions(myopts=None):
    """ Process command-line arguments."""
    description = (
        "This script runs a Kruscal-Wallis (KW) test on the rows of input data."
    )
    # Standard Input
    parser = argparse.ArgumentParser(
        description=description, formatter_class=RawDescriptionHelpFormatter
    )
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
        "--uniqueID",
        dest="uniqueID",
        action="store",
        required=True,
        help="Name of the column with unique identifiers.",
    )
    standard.add_argument(
        "-g",
        "--group",
        dest="group",
        action="store",
        required=True,
        default=False,
        help="Name of the column with groups.",
    )
    # Tool output
    output = parser.add_argument_group(title="Required output")
    output.add_argument(
        "-s",
        "--summaries",
        dest="summaries",
        action="store",
        required=True,
        help="Summaries file. TSV format.",
    )
    output.add_argument(
        "-f",
        "--flags",
        dest="flags",
        action="store",
        required=True,
        help="Flags file. TSV format.",
    )
    output.add_argument(
        "-v",
        "--volcano",
        dest="volcano",
        action="store",
        required=True,
        help="Volcano plot. PDF Format.",
    )
    # Plot Options
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

    # Standardized output paths
    args.input = os.path.abspath(args.input)
    args.design = os.path.abspath(args.design)
    args.summaries = os.path.abspath(args.summaries)
    args.flags = os.path.abspath(args.flags)
    args.volcano = os.path.abspath(args.volcano)

    return args


def main(args):
    """Main script function"""
    # Get data
    dat = wideToDesign(
        args.input, args.design, args.uniqueID, group=args.group, logger=logger
    )
    dat.wide = dat.wide.applymap(float)
    dat.dropMissing()
    # Getting the uinique pairs and all pairwise prermutations
    # to pass to Kruscal-Wallis.
    group_values_series_unique = dat.transpose()[dat.group].T.squeeze().unique()
    number_of_unique_groups = group_values_series_unique.shape[0]
    groups_pairwise = list(combinations(group_values_series_unique, 2))
    number_of_groups_pairwise = len(groups_pairwise)
    sample_ids = dat.wide.index.tolist()
    data_frame = dat.transpose()
    number_of_features = len(dat.wide.index)
    logger.info(f"Number of features: {number_of_features}")
    # Saving treatment group name from the arguments. Running overall Kruscall-Wallis test for all
    # group levels combined. Creating p_values_all and flag_values_all for 3 significance levels as
    # emply lists of length equal to the number_of_features. This will be used for all groups.
    p_value_all = [0] * number_of_features
    H_value_all = [0] * number_of_features
    mean_value_all = [0] * number_of_features
    variance_value_all = [0] * number_of_features
    flag_value_all_0p01 = [0] * number_of_features
    flag_value_all_0p05 = [0] * number_of_features
    flag_value_all_0p10 = [0] * number_of_features
    for j in range(0, number_of_features):
        # Creating a duplicate data frame for manipulation.
        data_frame_manipulate = data_frame
        # Dropping columns that characterize group. Only feature columns will remain.
        # We also trnaspose here so it will be easier to operate with.
        # group = 'White_wine_type_and_source'
        data_frame_manipulate.drop(args.group, 1)
        data_frame_manipulate = data_frame_manipulate.loc[:,sample_ids]
        data_frame_manipulate_transpose = data_frame_manipulate.transpose()
        # Pulling indexes list from the current data frame.
        indexes_list_complete = data_frame_manipulate_transpose.index.tolist()
        # Computing dataset summaries.
        input_values = data_frame_manipulate_transpose.loc[
            indexes_list_complete[j]
        ].values.astype(int)
        mean_value_all[j] = np.mean(input_values)
        variance_value_all[j] = np.var(
            data_frame_manipulate_transpose.loc[indexes_list_complete[j]].values, ddof=1
        )
        for i in range(0, number_of_unique_groups):
            # Extracting the pieces of the data frame that belong to ith unique group.
            data_frame_current_group = data_frame.loc[
                data_frame[args.group].isin([group_values_series_unique[i]])
            ]
            # Dropping columns that characterize group. Only feature columns will remain.
            # We also trnaspose here so it will be easier to operate with.
            data_frame_current_group.drop(args.group, 1)
            data_frame_current_group = data_frame_current_group.loc[:,sample_ids]
            data_frame_current_group = data_frame_current_group.transpose()
            # Pulling indexes list from the current data frame.
            indexes_list = data_frame_current_group.index.tolist()
            # Series current for group i and row (feature) j.
            series_current = data_frame_current_group.loc[indexes_list[j]]
            # This piece of code depends on whether it is the first group in the list or not.
            if i == 0:
                series_total = [series_current]
            else:
                series_total.append(series_current)
        # Checking if the compared elements are different.
        # Combining for checking.
        combined_list = data_frame_manipulate_transpose.loc[
            indexes_list_complete[j]
        ].tolist()
        combined_list_unique = np.unique(combined_list)
        # Checking if the number of unique elements is exactly 1.
        if len(combined_list_unique) == 1:
            # Performing Kruscal-Wallis for all groups for feature j.
            p_value_all[j] = float("nan")
            H_value_all[j] = float("nan")
            if p_value_all[j] < 0.01:
                flag_value_all_0p01[j] = 1
            if p_value_all[j] < 0.05:
                flag_value_all_0p05[j] = 1
            if p_value_all[j] < 0.10:
                flag_value_all_0p10[j] = 1
        else:
            # Performing Kruscal-Wallis for all groups for feature j.
            kruskal_wallis_input = [x.to_numpy() for x in series_total]
            p_value_all[j] = kruskalwallis(*kruskal_wallis_input)[1]
            H_value_all[j] = kruskalwallis(*kruskal_wallis_input)[0]
            if p_value_all[j] < 0.01:
                flag_value_all_0p01[j] = 1
            if p_value_all[j] < 0.05:
                flag_value_all_0p05[j] = 1
            if p_value_all[j] < 0.10:
                flag_value_all_0p10[j] = 1

    # The loop over features has to be finished by now. Converting them into the data frame.
    # The pariwise results will be added later.
    summary_df = pd.DataFrame(
        data=mean_value_all, columns=["GrandMean"], index=indexes_list
    )
    summary_df["SampleVariance"] = variance_value_all
    summary_df["H_value_for_all"] = H_value_all
    summary_df["prob_greater_than_H_for_all"] = p_value_all
    flag_df = pd.DataFrame(
        data=flag_value_all_0p01,
        columns=["flag_significant_0p01_on_all_groups"],
        index=indexes_list,
    )
    flag_df["flag_significant_0p05_on_all_groups"] = flag_value_all_0p05
    flag_df["flag_significant_0p10_on_all_groups"] = flag_value_all_0p10

    # Informing that KW for all group has been performed.
    logger.info("Kruscal-Wallis test for all groups together has been performed.")

    # Computing means for each group
    # This part just produces sumamry statistics for the output table.
    # This has nothing to do with Kruscal-Wallis

    for i in range(0, number_of_unique_groups):
        # Extracting the pieces of the data frame that belong to ith group.
        data_frame_current_group = data_frame.loc[
            data_frame[args.group].isin([group_values_series_unique[i]])
        ]

        # Dropping columns that characterize group. Only feature columns will remain.
        # We also trnaspose here so it will be easier to operate with.
        data_frame_current_group = data_frame_current_group.drop(
            args.group, 1
        ).transpose()
        # Pulling indexes list from the current group.
        indexes_list = data_frame_current_group.index.tolist()

        # Creating array of means for the current group that will be filled.
        means_value = [0] * number_of_features

        for j in range(0, number_of_features):

            series_current = data_frame_current_group.loc[indexes_list[j]]
            means_value[j] = series_current.mean()

        means_value_column_name_current = (
            "mean_treatment_" + group_values_series_unique[i]
        )
        summary_df[means_value_column_name_current] = means_value

    # Running pairwise Kruscall-Wallis test for all pairs of group levels that are saved in groups_pairwise.
    for i in range(0, number_of_groups_pairwise):
        # Extracting the pieces of the data frame that belong to groups saved in the i-th unique pair.
        groups_subset = groups_pairwise[i]
        data_frame_first_group = data_frame.loc[
            data_frame[args.group].isin([groups_subset[0]])
        ]
        data_frame_second_group = data_frame.loc[
            data_frame[args.group].isin([groups_subset[1]])
        ]

        # Dropping columns that characterize group. Only feature columns will remain.
        # We also trnaspose here so it will be easier to operate with.
        data_frame_first_group = data_frame_first_group.drop(args.group, 1).transpose()
        data_frame_second_group = data_frame_second_group.drop(
            args.group, 1
        ).transpose()
        # Pulling indexes list from the first one (they are the same)
        indexes_list = data_frame_first_group.index.tolist()

        # Creating p_values, neg_log10_p_value, flag_values, difference_value lists filled wiht 0es.
        p_value = [0] * number_of_features
        H_value = [0] * number_of_features
        neg_log10_p_value = [0] * number_of_features
        flag_value_0p01 = [0] * number_of_features
        flag_value_0p05 = [0] * number_of_features
        flag_value_0p10 = [0] * number_of_features
        difference_value = [0] * number_of_features

        for j in range(0, number_of_features):
            series_first = data_frame_first_group.loc[indexes_list[j]]
            series_second = data_frame_second_group.loc[indexes_list[j]]

            # Checking if the compared elements are different.
            # Combining for checking.
            first_list = data_frame_first_group.loc[indexes_list[j]].tolist()
            second_list = data_frame_second_group.loc[indexes_list[j]].tolist()
            combined_list = first_list + second_list
            combined_list_unique = np.unique(combined_list)
            # Checking if the number of unique elements is exactly 1.
            if len(combined_list_unique) == 1:
                p_value[j] = float("nan")
                H_value[j] = float("nan")
                # Possible alternative for two groups.
                # p_value[j] = kruskalwallis(series_first, series_second)[1]
                neg_log10_p_value[j] = -np.log10(p_value[j])
                difference_value[j] = series_first.mean() - series_second.mean()
                if p_value[j] < 0.01:
                    flag_value_0p01[j] = 1
                if p_value[j] < 0.05:
                    flag_value_0p05[j] = 1
                if p_value[j] < 0.10:
                    flag_value_0p10[j] = 1
            else:
                kruskal_wallis_args = [series_first, series_second]
                H_value[j], p_value[j] = kruskalwallis(kruskal_wallis_args[0].values, kruskal_wallis_args[1].values)
               #p_value[j], H_value[j] = kruskalwallis(kruskal_wallis_args[0].values, kruskal_wallis_args[1].values)
                # Possible alternative for two groups.
                # p_value[j] = kruskalwallis(series_first, series_second)[1]
                neg_log10_p_value[j] = -np.log10(p_value[j])
                difference_value[j] = series_first.mean() - series_second.mean()
                if p_value[j] < 0.01:
                    flag_value_0p01[j] = 1
                if p_value[j] < 0.05:
                    flag_value_0p05[j] = 1
                if p_value[j] < 0.10:
                    flag_value_0p10[j] = 1

        # Adding current p_value and flag_value column to the data frame and assigning the name
        p_value_column_name_current = (
            "prob_greater_than_H_for_diff_" + groups_subset[0] + "_" + groups_subset[1]
        )
        H_value_column_name_current = (
            "H_value_for_diff_" + groups_subset[0] + "_" + groups_subset[1]
        )
        neg_log10_p_value_column_name_current = (
            "neg_log10_p_value_" + groups_subset[0] + "_" + groups_subset[1]
        )
        difference_value_column_name_current = (
            "diff_of_" + groups_subset[0] + "_" + groups_subset[1]
        )
        summary_df[p_value_column_name_current] = p_value
        summary_df[H_value_column_name_current] = H_value
        summary_df[neg_log10_p_value_column_name_current] = neg_log10_p_value
        summary_df[difference_value_column_name_current] = difference_value

        flag_value_column_name_current_0p01 = (
            "flag_significant_0p01_on_" + groups_subset[0] + "_" + groups_subset[1]
        )
        flag_value_column_name_current_0p05 = (
            "flag_significant_0p05_on_" + groups_subset[0] + "_" + groups_subset[1]
        )
        flag_value_column_name_current_0p10 = (
            "flag_significant_0p10_on_" + groups_subset[0] + "_" + groups_subset[1]
        )
        flag_df[flag_value_column_name_current_0p01] = flag_value_0p01
        flag_df[flag_value_column_name_current_0p05] = flag_value_0p05
        flag_df[flag_value_column_name_current_0p10] = flag_value_0p10

    # Roundign the results up to 4 precison digits.
    summary_df = summary_df.apply(lambda x: x.round(4))

    # Adding name for the unique ID column that was there oroginally.
    summary_df.index.name = args.uniqueID
    flag_df.index.name = args.uniqueID

    # Save summary_df to the ouptut
    summary_df.to_csv(args.summaries, sep="\t")
    # Save flag_df to the output
    flag_df.to_csv(args.flags, sep="\t")

    # Informing that KW for pairwise group has been performed.
    logger.info("Kruscal-Wallis test for all groups pairwise has been performed.")

    # Generating Indexing for volcano plots.
    # Getting data for lpvals
    lpvals = {
        col.split("_value_")[-1]: summary_df[col]
        for col in summary_df.columns.tolist()
        if col.startswith("neg_log10_p_value")
    }

    # Gettign data for diffs
    difs = {
        col.split("_of_")[-1]: summary_df[col]
        for col in summary_df.columns.tolist()
        if col.startswith("diff_of_")
    }

    # The cutoff value for significance.
    cutoff = 2

    # Making volcano plots
    with PdfPages(args.volcano) as pdf:
        for i in range(0, number_of_groups_pairwise):
            # Set Up Figure
            volcanoPlot = figureHandler(proj="2d")

            groups_subset = groups_pairwise[i]
            current_key = groups_subset[0] + "_" + groups_subset[1]

            # Plot all results
            scatter.scatter2D(
                x=list(difs[current_key]),
                y=list(lpvals[current_key]),
                colorList=list("b"),
                ax=volcanoPlot.ax[0],
            )

            # Color results beyond treshold red
            cutLpvals = lpvals[current_key][lpvals[current_key] > cutoff]
            if not cutLpvals.empty:
                cutDiff = difs[current_key][cutLpvals.index]
                scatter.scatter2D(
                    x=list(cutDiff),
                    y=list(cutLpvals),
                    colorList=list("r"),
                    ax=volcanoPlot.ax[0],
                )

            # Drawing cutoffs
            lines.drawCutoffHoriz(y=cutoff, ax=volcanoPlot.ax[0])

            # Format axis (volcanoPlot)
            volcanoPlot.formatAxis(
                axTitle=current_key,
                grid=False,
                yTitle="-log10(p-value) for Diff of treatment means for {0}".format(
                    current_key
                ),
                xTitle="Difference of treatment means for {0}".format(current_key),
            )

            # Add figure to PDF
            volcanoPlot.addToPdf(pdfPages=pdf)

    # Informing that the volcano plots are done
    logger.info("Pairwise volcano plots have been created.")

    logger.info("Finishing running of Kruscal-Wallis tests.")


if __name__ == "__main__":
    args = getOptions()
    logger = logging.getLogger()
    sl.setLogger(logger)
    logger.info(
        """Importing data with following parameters:
                Input: {0}
                Design: {1}
                uniqueID: {2}
                group: {3}
                """.format(
            args.input, args.design, args.uniqueID, args.group
        )
    )
    palette = colorHandler(pal=args.palette, col=args.color)
    logger.info(
        "Using {0} color scheme from {1} palette".format(args.color, args.palette)
    )
    warnings.filterwarnings("ignore")
    main(args)
