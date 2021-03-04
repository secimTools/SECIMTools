#!/usr/bin/env python
######################################################################################
# AUTHOR: Alexander Kirpich <akirpich@ufl.edu>
#
# DESCRIPTION: This tool runs t-test which can be either:
#                   "paired"   (for two groups, pairingID has to be provided) or
#                   "unpaired" (pairwise for all groups).
#######################################################################################

from __future__ import division
import os
import logging
import argparse
import warnings
from itertools import combinations
from argparse import RawDescriptionHelpFormatter
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from secimtools.visualManager import module_lines as lines
from secimtools.visualManager import module_scatter as scatter
from secimtools.visualManager.manager_color import colorHandler
from secimtools.visualManager.manager_figure import figureHandler
from secimtools.dataManager import logger as sl
from secimtools.dataManager.interface import wideToDesign
import matplotlib

# Set a non-interactive backend for running on a cluster
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages


def getOptions(myopts=None):
    """ Parse arguments """
    # Standard Input
    parser = argparse.ArgumentParser(
        description="Run an [un]paired t-test on all features",
        formatter_class=RawDescriptionHelpFormatter,
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
        help="Name of the column with group variable.",
    )
    standard.add_argument(
        "-p",
        "--pairing",
        dest="pairing",
        action="store",
        required=True,
        choices=['paired', 'unpaired'],
        default='unpaired',
        help="T-Test type, paired or unpaired.",
    )
    standard.add_argument(
        "-o",
        "--order",
        dest="order",
        action="store",
        required=False,
        default=False,
        help="Name of the pairing vaiable if paired option is selected. "
        "Ignored when unpaired test is selected.",
    )
    # Tool output
    output = parser.add_argument_group(title="Required output")
    output.add_argument(
        "-s",
        "--summaries",
        dest="summaries",
        action="store",
        required=True,
        help="Summaries file name. TSV format.",
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
        help="Volcano plot output PDF file.",
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


def create_volcano_plots(args, summary_df, groups_pairwise, number_of_groups_pairwise, difs, lpvals):
    """Create volcano plots and generate a PDF file as specified."""
    # Making volcano plots
    cutoff = 2
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

            # Color results beyond threshold red
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


def run_unpaired_ttest(args, dat):
    """Perform an unpaired T-Test on the provided data."""
    order = args.order
    logger.info("Unpaired t-test will be performed for all groups pairwise.")
    # Get the unique pairs and all pairwise prermutations
    # to feed them pairwise to unpaired t-tests.
    group_values_series = dat.transpose()[dat.group].T.squeeze()
    group_values_series_unique = group_values_series.unique()
    number_of_unique_groups = group_values_series_unique.shape[0]
    groups_pairwise = list(combinations(group_values_series_unique, 2))
    number_of_groups_pairwise = len(groups_pairwise)
    # Extracting data from the interface. Only extract data columns.
    sample_ids = dat.wide.index.tolist()
    data_frame = dat.transpose() # Extracting number of features. This will depend on whether the user has provided
    # an ordering variable or not.
    # run_Order_Fake_variable:
    # This variable is useless for an unpared test. It just adds extra column to data frame.
    number_of_features = len(sample_ids)
    # Saving treatment group name from the arguments.
    # Computing overall summaries (mean and variance).
    # This part just produces sumamry statistics for the output table.
    # This has nothing to do with unpaired t-test. This is just summary for the table.
    mean_value_all = [0] * number_of_features
    variance_value_all = [0] * number_of_features
    # Creating duplicate for manipulation.
    data_frame_manipulate = data_frame.loc[:,sample_ids]
    data_frame_manipulate_transpose = data_frame_manipulate.transpose()
    # Pull index list from the current data frame.
    indexes_list_complete = data_frame_manipulate_transpose.index.tolist()

    # Compute mean and variance per feature
    for j in range(0, number_of_features):
        complete_list = data_frame_manipulate_transpose.loc[indexes_list_complete[j]]
        mean_value_all[j] = np.mean(complete_list)
        variance_value_all[j] = np.var(
            data_frame_manipulate_transpose.loc[indexes_list_complete[j]], ddof=1
        )

    # Create the results table
    summary_df = pd.DataFrame(
        data=mean_value_all, columns=["GrandMean"], index=indexes_list_complete
    )
    summary_df["SampleVariance"] = variance_value_all

    # Compute means for each group and outpute them.
    # Produce summary statistics for the output table first.
    for i in range(0, number_of_unique_groups):
        # Extract the i(th) group.
        data_frame_current_group = data_frame.loc[
            data_frame[args.group].isin([group_values_series_unique[i]])
        ]

        # Select feature columns and transpose the resulting df
        data_frame_current_group = data_frame_current_group.loc[:,sample_ids].transpose()

        # Pull indexes from the current group.
        indexes_list = data_frame_current_group.index.tolist()

        # Create empty array of means for the current group
        means_value = [0] * number_of_features

        for j in range(0, number_of_features):
            series_current = data_frame_current_group.loc[indexes_list[j]]
            means_value[j] = series_current.mean()

        # Add current mean_value column to the data frame and assign the name.
        means_value_column_name_current = (
            "mean_treatment_" + group_values_series_unique[i]
        )
        summary_df[means_value_column_name_current] = means_value

    # Run a pairwise unpaired (two-sample) t-test for all pairs of group levels that are
    # saved in groups_pairwise.
    ttest_df = data_frame.loc[:,sample_ids]
    for i in range(0, number_of_groups_pairwise):
        # Extract the group data in the i-th unique pair.
        groups_subset = groups_pairwise[i]
        data_frame_first_group = ttest_df.loc[
            data_frame[args.group].isin([groups_subset[0]])
        ].transpose()
        data_frame_second_group = ttest_df.loc[
            data_frame[args.group].isin([groups_subset[1]])
        ].transpose()

        # Pull indexes list from the first group as both groups are the same
        indexes_list = data_frame_first_group.index.tolist()

        # Create p_values, neg_log10_p_value, flag_values, difference_value lists filled with 0s
        p_value = [0] * number_of_features
        t_value = [0] * number_of_features
        neg_log10_p_value = [0] * number_of_features
        flag_value_0p01 = [0] * number_of_features
        flag_value_0p05 = [0] * number_of_features
        flag_value_0p10 = [0] * number_of_features
        difference_value = [0] * number_of_features

        for j in range(0, number_of_features):
            series_first = data_frame_first_group.loc[indexes_list[j]]
            series_second = data_frame_second_group.loc[indexes_list[j]]
            ttest_ind_args = [series_first, series_second]
            p_value[j] = ttest_ind(*ttest_ind_args)[1]
            t_value[j] = ttest_ind(*ttest_ind_args)[0]
            # Possible alternative for two groups.
            # p_value[j] = ttest_ind_args(series_first, series_second)[1]
            neg_log10_p_value[j] = -np.log10(p_value[j])
            difference_value[j] = series_first.mean() - series_second.mean()
            if p_value[j] < 0.01:
                flag_value_0p01[j] = 1
            if p_value[j] < 0.05:
                flag_value_0p05[j] = 1
            if p_value[j] < 0.10:
                flag_value_0p10[j] = 1

        # Create column names for the data frame.
        p_value_column_name_current = (
            "prob_greater_than_t_for_diff_" +
            groups_subset[0] +
            "_" +
            groups_subset[1]
        )
        t_value_column_name_current = (
            "t_value_for_diff_" + groups_subset[0] + "_" + groups_subset[1]
        )
        neg_log10_p_value_column_name_current = (
            "neg_log10_p_value_" + groups_subset[0] + "_" + groups_subset[1]
        )
        difference_value_column_name_current = (
            "diff_of_" + groups_subset[0] + "_" + groups_subset[1]
        )
        flag_value_column_name_current_0p01 = (
            "flag_significant_0p01_on_" + groups_subset[0] + "_" + groups_subset[1]
        )
        flag_value_column_name_current_0p05 = (
            "flag_significant_0p05_on_" + groups_subset[0] + "_" + groups_subset[1]
        )
        flag_value_column_name_current_0p10 = (
            "flag_significant_0p10_on_" + groups_subset[0] + "_" + groups_subset[1]
        )

        # Adding current p_value and flag_value column to the data frame and assigning the name.
        # If the data frame has not been created (i == 0) then create it on the fly.
        if i == 0:
            flag_df = pd.DataFrame(
                data=flag_value_0p01,
                columns=[flag_value_column_name_current_0p01],
                index=indexes_list,
            )
        else:
            flag_df[flag_value_column_name_current_0p01] = flag_value_0p01

        # At this point data frame exists so only columns are added to the existing data frame.
        summary_df[p_value_column_name_current] = p_value
        summary_df[t_value_column_name_current] = t_value
        summary_df[neg_log10_p_value_column_name_current] = neg_log10_p_value
        summary_df[difference_value_column_name_current] = difference_value
        flag_df[flag_value_column_name_current_0p05] = flag_value_0p05
        flag_df[flag_value_column_name_current_0p10] = flag_value_0p10
    return(summary_df, flag_df, groups_pairwise, number_of_groups_pairwise)


def run_paired_ttest(args, dat):
    """Perform a paired T-Test on the provided data."""
    order = args.order
    logger.info(f"Pairwise t-test based on a pairing variable: {order}.".format(order))

    # Get the number of unique groups. If it is bigger than 2 return an error.
    group_values_series = dat.transpose()[dat.group].T.squeeze()
    group_values_series_unique = group_values_series.unique()
    print(group_values_series_unique)
    number_of_unique_groups = group_values_series_unique.shape[0]
    if number_of_unique_groups != 2:
        logger.warning(
            "Expected 2 unique groups, received {0}. Cannot perform a paired t-test.".format(
                number_of_unique_groups
            )
        )
        exit()

    # This piece of code will be executed only if the number_of_unique_groups is exactly 2 so
    # the group check is passed. Creating pairwise combination of our two groups that we will
    # use in the future.
    groups_pairwise = list(combinations(group_values_series_unique, 2))
    print(groups_pairwise)
    number_of_groups_pairwise = len(groups_pairwise)

    # Extracting data from the interface.
    data_frame = dat.transpose()
    # Extracting number of features. This will depend on whether the user has provided ordering
    # variable or not. Checking that the requred pairing variable has been provided.
    if args.order is False:
        logger.info(
            "Required pairing variable not provided: Paired t-test cannot be performed."
        )
        exit()

    # This piece of code will be executed only if the args.order has been provided and the
    # check is passed. Defining the number of features. It should be the dimension of the data
    # frame minus 2 columns that stand for arg.group and args.order
    number_of_features = data_frame.shape[1] - 2

    # At this point is is confirmed that there are only 2 groups and that pairing variable
    # args.order has been provided. Now we need to check that pairing is correct i.e. that each
    # pairID corresponds to only two samples from different groups. Getting the unique pairs
    # and deleting those theat have more or less than three.
    pairid_values_series = dat.transpose()[dat.runOrder].T.squeeze()
    pairid_values_series_unique = pairid_values_series.unique()
    number_of_unique_pairid = pairid_values_series_unique.shape[0]

    # Extracting data from the interface.
    data_frame = dat.transpose()

    # Performing the cleaning of the original data. We are removing samples that are not paired
    # and not belonging to the two groups. If the dataset has 1 or 3 or more matches for a
    # pairid those samples are removed with a warning. If pairid corresponds to exactly two
    # samples (which is correct) but groupid values are NOT different those values will be also
    # removed.
    for i in range(0, number_of_unique_pairid):
        # Extracting the pieces of the data frame that belong to ith unique pairid.
        data_frame_current_pairid = data_frame.loc[
            data_frame[args.order].isin([pairid_values_series_unique[i]])
        ]

        # We transpose here so it will be easier to operate with.
        data_frame_current_pairid = data_frame_current_pairid.transpose()
        sample_names_current_pairid = list(data_frame_current_pairid.columns.values)
        if data_frame_current_pairid.shape[1] != 2:
            # Pulling indexes list from the current data frame.
            logger.warning(
                """Number of samples for pairID must be equal to 2, not {0}. Sample(s) {1} will
                   be removed from further analysis.""".format(
                    data_frame_current_pairid.shape[1],
                    sample_names_current_pairid,
                )
            )

            # Getting indexes we are trying to delete.
            boolean_indexes_to_delete = data_frame.index.isin(
                sample_names_current_pairid
            )
            # Deleting the indexes and in the for loop going to next iteration.
            data_frame.drop(
                data_frame.index[boolean_indexes_to_delete], inplace=True
            )

        # This piece is executed if the numbe is correct i.e. data_frame_current_group.shape[1]
        # == 2: Here we are checking if the groupID-s for the given pair are indeed different.

        elif (
            data_frame_current_pairid.transpose()[args.group][0] ==
            data_frame_current_pairid.transpose()[args.group][1]
        ):
            logger.warning(
                """Samples in pairID {0} have groupIDs: {1} and {2}. They should be different!
                   Sample(s) {3} will be removed from further analysis.""".format(
                    pairid_values_series_unique[i],
                    data_frame_current_pairid.transpose()[args.group][1],
                    data_frame_current_pairid.transpose()[args.group][0],
                    sample_names_current_pairid,
                )
            )
            # Getting indexes we are trying to delete.
            boolean_indexes_to_delete = data_frame.index.isin(
                sample_names_current_pairid
            )
            # Deleting the indexes.
            data_frame.drop(
                data_frame.index[boolean_indexes_to_delete], inplace=True
            )

    # Checking if the data frame became empty after cleaning.
    if data_frame.shape[0] == 0:
        logger.warning(
            """Number of paired samples in the final dataset is 0!
               Please check the desing file for accuracy! Exiting the program."""
        )
        exit()

    # Computing overall summaries (mean and variance).
    # This part just produces sumamry statistics for the output table.
    # This has nothing to do with paired t-test. This is just summary for the table.
    mean_value_all = [0] * number_of_features
    variance_value_all = [0] * number_of_features

    for j in range(0, number_of_features):
        # Creating duplicate for manipulation.
        data_frame_manipulate = data_frame

        # Dropping columns that characterize group. Only feature columns will remain.
        # We also trnaspose here so it will be easier to operate with.
        data_frame_manipulate_transpose = data_frame_manipulate.drop(
            [args.group, args.order], 1
        ).transpose()
        # Pulling indexes list from the current data frame.
        indexes_list_complete = data_frame_manipulate_transpose.index.tolist()

        # Computing dataset summaries.
        mean_value_all[j] = np.mean(
            data_frame_manipulate_transpose.loc[indexes_list_complete[j]]
        )
        variance_value_all[j] = np.var(
            data_frame_manipulate_transpose.loc[indexes_list_complete[j]], ddof=1
        )

    # Creating the table and putting the results there.
    summary_df = pd.DataFrame(
        data=mean_value_all, columns=["GrandMean"], index=indexes_list_complete
    )
    summary_df["SampleVariance"] = variance_value_all

    # Computing means for each group and outputting them.
    # This part just produces summary statistics for the output table.
    # This has nothing to do with paired t-test. This is just summary for the table.

    for i in range(0, number_of_unique_groups):
        # Extracting the pieces of the data frame that belong to the ith group.
        data_frame_current_group = data_frame.loc[
            data_frame[args.group].isin([group_values_series_unique[i]])
        ]

        # Dropping columns that characterize group. Only feature columns will remain.
        data_frame_current_group = data_frame_current_group.drop(
            [args.group, args.order], 1
        ).transpose()

        # Pulling indexes list from the current group.
        indexes_list = data_frame_current_group.index.tolist()

        # Creating array of means for the current group that will be filled.
        means_value = [0] * number_of_features

        for j in range(0, number_of_features):
            series_current = data_frame_current_group.loc[indexes_list[j]]
            means_value[j] = series_current.mean()

        # Adding current mean_value column to the data frame and assigning the name.
        means_value_column_name_current = (
            "mean_treatment_" + group_values_series_unique[i]
        )
        summary_df[means_value_column_name_current] = means_value

    # Performing paired t-test for the two groups and saving results.
    # Creating p_values and flag_values empty list of length number_of_features.
    # This will be used for the two groups in paired t-test.
    p_value = [0] * number_of_features
    t_value = [0] * number_of_features
    flag_value_0p01 = [0] * number_of_features
    flag_value_0p05 = [0] * number_of_features
    flag_value_0p10 = [0] * number_of_features
    neg_log10_p_value = [0] * number_of_features
    difference_value = [0] * number_of_features

    # Performing paired t-test for each pair of features.
    for j in range(0, number_of_features):
        # Extracting the pieces of the data frame that belong to 1st group.
        data_frame_first_group = data_frame.loc[
            data_frame[args.group].isin([group_values_series_unique[0]])
        ]
        data_frame_second_group = data_frame.loc[
            data_frame[args.group].isin([group_values_series_unique[1]])
        ]

        # Sorting data frame by args.group index
        # This will ensure datasets are aligned by pair when fed to the t-test.
        data_frame_first_group = data_frame_first_group.sort_values(args.order)
        data_frame_second_group = data_frame_second_group.sort_values(args.order)

        # Sorting data frame by args.group index
        data_frame_first_group = data_frame_first_group.drop(
            [args.group, args.order], 1
        ).transpose()
        data_frame_second_group = data_frame_second_group.drop(
            [args.group, args.order], 1
        ).transpose()

        # Pulling list of indexes. This is the same list for the first and for the second.
        indexes_list = data_frame_first_group.index.tolist()

        # Pullinng the samples out
        series_first = data_frame_first_group.loc[indexes_list[j]]
        series_second = data_frame_second_group.loc[indexes_list[j]]

        # Running t-test for the two given samples
        paired_ttest_args = [series_first, series_second]
        p_value[j] = ttest_rel(*paired_ttest_args)[1]
        t_value[j] = ttest_rel(*paired_ttest_args)[0]
        neg_log10_p_value[j] = -np.log10(p_value[j])
        difference_value[j] = series_first.mean() - series_second.mean()
        if p_value[j] < 0.01:
            flag_value_0p01[j] = 1
        if p_value[j] < 0.05:
            flag_value_0p05[j] = 1
        if p_value[j] < 0.10:
            flag_value_0p10[j] = 1

    # The loop over features has to be finished by now. Converting them into the data frame.
    # Creating column names for the data frame.
    p_value_column_name_current = (
        "prob_greater_than_t_for_diff_" +
        group_values_series_unique[0] +
        "_" +
        group_values_series_unique[1]
    )
    t_value_column_name_current = (
        "t_value_for_diff_" +
        group_values_series_unique[0] +
        "_" +
        group_values_series_unique[1]
    )
    neg_log10_p_value_column_name_current = (
        "neg_log10_p_value_" +
        group_values_series_unique[0] +
        "_" +
        group_values_series_unique[1]
    )
    difference_value_column_name_current = (
        "diff_of_" +
        group_values_series_unique[0] +
        "_" +
        group_values_series_unique[1]
    )
    flag_value_column_name_current_0p01 = (
        "flag_value_diff_signif_" +
        group_values_series_unique[0] +
        "_" +
        group_values_series_unique[1] +
        "_0p01"
    )
    flag_value_column_name_current_0p05 = (
        "flag_value_diff_signif_" +
        group_values_series_unique[0] +
        "_" +
        group_values_series_unique[1] +
        "_0p05"
    )
    flag_value_column_name_current_0p10 = (
        "flag_value_diff_signif_" +
        group_values_series_unique[0] +
        "_" +
        group_values_series_unique[1] +
        "_0p10"
    )

    summary_df[t_value_column_name_current] = t_value
    summary_df[p_value_column_name_current] = p_value
    summary_df[neg_log10_p_value_column_name_current] = neg_log10_p_value
    summary_df[difference_value_column_name_current] = difference_value

    flag_df = pd.DataFrame(
        data=flag_value_0p01,
        columns=[flag_value_column_name_current_0p01],
        index=indexes_list,
    )
    flag_df[flag_value_column_name_current_0p05] = flag_value_0p05
    flag_df[flag_value_column_name_current_0p10] = flag_value_0p10
    return(summary_df, flag_df, groups_pairwise, number_of_groups_pairwise)


def main(args):
    """Main body of the script"""
    logger.info("Loading data with the Interface")
    dat = wideToDesign(
        args.input,
        args.design,
        args.uniqueID,
        group=args.group,
        runOrder=args.order,
        logger=logger,
    )
    # Treat everything as numeric
    dat.wide = dat.wide.applymap(float)

    # Drop missing data!
    dat.dropMissing()

    # SCENARIO 1: Unpaired t-test. In this case there can be as many groups as possible.
    # Order variable is ignored and t-tests are performed pairwise for each pair of groups.
    if args.pairing == "unpaired":
        summary_df, flag_df, groups_pairwise, number_of_groups_pairwise = run_unpaired_ttest(args, dat)

    # SCENARIO 2: Paired t-test. In this case there should be EXACTLY TWO groups.
    # Each sample in one group should have exacty one matching pair in the other group.
    # The matching is controlled by args.order variable.
    elif args.pairing == "paired":
        summary_df, flag_df, groups_pairwise, number_of_groups_pairwise = run_paired_ttest(args, dat)

    # Roundign the results up to 4 precision digits.
    summary_df = summary_df.apply(lambda x: x.round(4))

    # Adding name for the unique ID column that was there oroginally.
    summary_df.index.name = args.uniqueID
    flag_df.index.name = args.uniqueID

    # Save summary_df to the ouptut
    summary_df.to_csv(args.summaries, sep="\t")
    # Save flag_df to the output
    flag_df.to_csv(args.flags, sep="\t")

    # ### Volcano Plots
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

    create_volcano_plots(args, summary_df, groups_pairwise, number_of_groups_pairwise, difs, lpvals)

    # Ending script
    logger.info("Finishing t-test run.")


if __name__ == "__main__":
    args = getOptions()
    logger = logging.getLogger()
    sl.setLogger(logger)
    logger.info(
        """Importing data with following parameters:
                Input: {0}
                Design: {1}
                UniqueID: {2}
                Group: {3}
                TestType: {4}
                pairID: {5}
                """.format(
            args.input, args.design, args.uniqueID, args.group, args.pairing, args.order
        )
    )
    palette = colorHandler(pal=args.palette, col=args.color)
    logger.info(
        "Using {0} color scheme from {1} palette".format(args.color, args.palette)
    )
    warnings.filterwarnings("ignore")
    main(args)
