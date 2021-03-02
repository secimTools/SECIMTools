#!/usr/bin/env python
################################################################################
# AUTHORS: Miguel A Ibarra (miguelib@ufl.edu)
#          Alexander Kirpich (akirpich@ufl.edu)
#          Matt Thoburn (mthoburn@ufl.edu)
#
# DESCRIPTION: This script takes a a wide format file and makes a partial
#               least squares discriminant analysis (PLS-DA).
################################################################################

import os
import logging
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
from argparse import RawDescriptionHelpFormatter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from secimtools.dataManager import logger as sl
from secimtools.dataManager.interface import wideToDesign
from secimtools.visualManager import module_scatter as scatter
from secimtools.visualManager.manager_color import colorHandler
from secimtools.visualManager.manager_figure import figureHandler


def getOptions(myopts=None):
    """ Function to pull in arguments """
    description = """
    This script performs Partial Least Squares Discriminant Analysis (PLS-DA) for the selected groups

    """
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
        required=True,
        default=False,
        help="Name of the column" " with groups.",
    )
    standard.add_argument(
        "-l",
        "--levels",
        dest="levels",
        action="store",
        required=False,
        default=False,
        help="Different groups to" " sort by separeted by commas.",
    )
    tool = parser.add_argument_group(
        title="Tool specific input", description="Input specific for this tool."
    )
    tool.add_argument(
        "-t",
        "--toCompare",
        dest="toCompare",
        action="store",
        required=True,
        default=True,
        help="Name of" " the elements to compare in group col.",
    )
    tool.add_argument(
        "-cv",
        "--cross_validation",
        dest="cross_validation",
        action="store",
        required=True,
        help="Choice of cross-validation procedure for the -nc determinantion: none, "
        "single, double.",
    )
    tool.add_argument(
        "-n",
        "--nComp",
        dest="nComp",
        action="store",
        required=False,
        default=2,
        type=int,
        help="Number" " of components.",
    )
    output = parser.add_argument_group(title="Required output")
    output.add_argument(
        "-os",
        "--outScores",
        dest="outScores",
        action="store",
        required=True,
        help="Name of output file to store loadings. TSV format.",
    )
    output.add_argument(
        "-ow",
        "--outWeights",
        dest="outWeights",
        action="store",
        required=True,
        help="Name of output file to store weights. TSV format.",
    )
    output.add_argument(
        "-oc",
        "--outClassification",
        dest="outClassification",
        action="store",
        required=True,
        help="Name of output file to store classification. TSV format.",
    )
    output.add_argument(
        "-oca",
        "--outClassificationAccuracy",
        dest="outClassificationAccuracy",
        action="store",
        required=True,
        help="Name of output file to store classification accuracy. TSV format.",
    )
    output.add_argument(
        "-f",
        "--figure",
        dest="figure",
        action="store",
        required=False,
        help="Name of output file to store scatter plots for scores",
    )
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
    development = parser.add_argument_group(title="Development Settings")
    development.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        required=False,
        help="Add debugging log output.",
    )
    args = parser.parse_args()

    # Standardize Paths
    args.input = os.path.abspath(args.input)
    args.design = os.path.abspath(args.design)
    args.figure = os.path.abspath(args.figure)
    args.outScores = os.path.abspath(args.outScores)
    args.outWeights = os.path.abspath(args.outWeights)
    args.outClassification = os.path.abspath(args.outClassification)
    args.outClassificationAccuracy = os.path.abspath(args.outClassificationAccuracy)

    # Split levels if levels
    if args.levels:
        args.levels = args.levels.split(",")

    # Split to compare
    args.toCompare = args.toCompare.split(",")

    return args


def runPLS(trans, group, toCompare, nComp, cv_status):
    """
    Runs PCA over a wide formated dataset

    :Arguments:
        :type dat: pandas.core.frame.DataFrame
        :param dat: DataFrame with the wide file data

    :Returns:
        :rtype scores: pandas.DataFrame
        :return scores: Scores of the PCA

        :rtype weights: pandas.DataFrame
        :return weights: weights of the PCA
    """
    logger.info("Runing PLS on data")

    subset = pd.DataFrame
    print(f"Groups to compare: {toCompare}")
    subset = [subs for name, subs in trans.groupby(group) if name in toCompare]
    subset = pd.concat(subset, axis=0)

    # Creating pseudolinear Y value
    Y = np.where(subset[group] == toCompare[0], int(1), int(0))

    # Remove group column from the subset data
    subset.drop(group, axis=1, inplace=True)

    # Check the status from the cross-validation procedure
    # The code below depends on cross validation status. The status can be either "single", "double" or "none".

    # Case 1: User provides cv_status = "none". No cross-validation will be performed.
    # The number of components shoul be specified by the user in the nComp variable. The default in xml shoul be 2.
    if cv_status == "none":

        # Telling the user that we are using the number of components pre-specified by the user.
        logger.info("Using the number of components specified by the user.")

        # If by mistake the number the componentes nComp was not provided we hardcode it to 2.
        if nComp is None:
            logger.info(
                "The number of componets was not provided! Default number 2 is used."
            )
            nComp = 2

        # Putting the user defined (or corrected if nothing imputted) number of compoents nComp directly into index_min.
        index_min = nComp

    # Case 2: User provides cv_status = "single". Only single cross-validation will be performed.
    if cv_status == "single":

        # Telling the user that we are using the number of components determined via a single cross-validation.
        logger.info(
            "Using the number of components determined via a single cross-validation."
        )

        # Checking if the sample sizes is smaller than 100 and exiting if that is the case.
        if len(Y) < 100:
            logger.info(
                "The required number of samples for a single cross-validation procedure is at least 100. The dataset has {0}.".format(
                    len(Y)
                )
            )
            logger.info("Exiting the tool.")
            exit()

        # Pulling the maximum number of components that will be used for cross-validation.
        n_max = subset.shape[0]

        # Creating a list of values to perform single cross-validation over.
        # P.S. We do not consider scenario of a single component so that we can produce at least single 2D plot in the end.
        n_list = list(range(2, n_max + 1))

        # Creating dictionary we gonna feed to the single cross-validation procedure.
        n_list_dictionary = dict(n_components=n_list)

        # Creating a gridsearch object with parameter "n_list_dictionary"
        internal_cv = GridSearchCV(
            estimator=PLSRegression(), param_grid=n_list_dictionary
        )

        # Performing internal_cv.
        internal_cv.fit(subset.values, Y)

        index_min = internal_cv.best_params_["n_components"]

    # Case 3: User provides cv_status = "double". Double cross-validation will be performed.
    if cv_status == "double":

        # Telling the user that we are using the number of components determined via a double cross-validation.
        logger.info(
            "Using the number of components determined via a double cross-validation."
        )

        # Checking if the sample sizes is smaller than 100 and exiting if that is the case.
        if len(Y) < 100:
            logger.info(
                "The required number of samples for a double cross-validation procedure is at least 100. The dataset has {0}.".format(
                    len(Y)
                )
            )
            logger.info("Exiting the tool.")
            exit()

        # Pulling the maximum number of components that will be used for cross-validation.
        n_max = subset.shape[0]

        # Here we are looping over possible lists we have to cross-validate over.
        # We do not consider scenario of a single components so that we can produce at least one 2D plot in the end.

        # Creating index of the minimum variable that will give us the best prediction.
        # This will be updated during interlan and external CV steps if necessary.
        index_min = 2

        for n_current in range(2, n_max + 1):

            # Creating the set of candidates that we will use for both cross-validation loops: internal and external
            n_list = list(range(2, n_current + 1))

            # Creating dictionary we gonna feed to the internal cross-validation procedure.
            n_list_dictionary = dict(n_components=n_list)

            # Creating a gridsearch object with parameter "n_list_dictionary"
            internal_cv = GridSearchCV(
                estimator=PLSRegression(), param_grid=n_list_dictionary
            )

            # Performing internal_cv.
            # internal_cv.fit( subset.values, Y )

            # Performing external_cv using internal_cv
            external_cv = cross_val_score(internal_cv, subset.values, Y)

            # Checking whether adding this extra component to our anlaysis will help.
            # For the first 2 components we assume they are the best and update it later if necessary.
            if n_current == 2:
                best_predction_proportion = external_cv.mean()

            else:
                # Checking whether adding this extra component helped to what we already had.
                if external_cv.mean() > best_predction_proportion:
                    best_predction_proportion = external_cv.mean()
                    index_min = n_current

    # Apply PLS-DA model to our data, we neet to specify the number of latent
    # variables (LV), to use for our data.
    # index_min was determined above either via cross-validantion or by itself.
    plsr = PLSRegression(n_components=index_min)

    # Fit the model
    plsr.fit(subset.values, Y)

    # Creating column names for scores and weights
    LVNames = ["LV_{0}".format(i + 1) for i in range(index_min)]

    # The scores describe the position of each sample in each determined LV.
    # Columns for LVs and rows for Samples. (These are the ones the you plot).
    # Generrates a dataFrame with the scores.
    scores_df = pd.DataFrame(plsr.x_scores_, index=subset.index, columns=LVNames)

    # The weights describe the contribution of each variable to each LV. The
    # columns are for the LVs and the rows are for each feature.
    # generates a dataFrame with the weights.
    weights_df = pd.DataFrame(plsr.x_weights_, index=subset.columns, columns=LVNames)

    # Dealing with predicted and classification data frame.

    # Geting predicted values of PLS (Fit PLS)
    fitted_values = plsr.predict(subset.values)
    fitted_values_round = fitted_values.round()

    # Combining results into the data_frame so that it can be exported.
    classification_df = pd.DataFrame(
        {
            "Group_Observed": Y,
            "Group_Predicted": fitted_values.T.squeeze(),
            "Group_Predicted_Rounded": fitted_values_round.T.squeeze(),
        },
        index=subset.index,
    )

    # Returning the results
    return scores_df, weights_df, classification_df


def plotScores(data, palette, pdf):
    """
    This function creates a PDF file with 3 scatter plots for the combinations
    of the 3 principal components. PC1 vs PC2, PC1 vs PC3, PC2 vs PC3.

    :Arguments:
        :type data: pandas.core.frame.DataFrame
        :param data: Data frame with the data to plot.

        :type outpath: string
        :param outpath: Path for the output file

        :type group: string
        :param group: Name of the column that contains the group information on the design file.

    :Return:
        :rtype PDF: file
        :retrn PDF: file with the 3 scatter plots for PC1 vs PC2, PC1 vs PC3, PC2  vs PC3.
    """
    for x, y in list(itertools.combinations(data.columns.tolist(), 2)):
        # Creating a figure handler object
        fh = figureHandler(proj="2d", figsize=(14, 8))

        # Creating title for the figure
        title = "{0} vs {1}".format(x, y)

        # Creating the scatterplot 2D
        scatter.scatter2D(
            ax=fh.ax[0],
            x=list(data[x]),
            y=list(data[y]),
            colorList=palette.design.colors.tolist(),
        )

        # Despine axis
        fh.despine(fh.ax[0])

        fh.makeLegend(ax=fh.ax[0], ucGroups=palette.ugColors, group=palette.combName)

        # Shinking the plot so everything fits
        fh.shrink()

        # Format Axis
        fh.formatAxis(
            figTitle=title,
            xTitle="Scores on {0}".format(x),
            yTitle="Scores on {0}".format(y),
            grid=False,
        )

        # Adding figure to pdf
        fh.addToPdf(dpi=90, pdfPages=pdf)


def main(args):
    """Main function"""
    if args.levels and args.group:
        levels = [args.group] + args.levels
    elif args.group and not args.levels:
        levels = [args.group]
    else:
        levels = []

    dat = wideToDesign(
        args.input,
        args.design,
        args.uniqID,
        group=args.group,
        anno=args.levels,
        logger=logger,
    )

    # Treat everything as numeric
    dat.wide = dat.wide.applymap(float)

    # Cleaning from missing data
    dat.dropMissing()

    # Get remaining Sample IDs for dataframe filtering
    sample_ids = dat.wide.index.tolist()
    # Re-add the group variable needed for downstream analysis
    sample_ids.append(dat.group)

    # Get colors for each sample based on the group
    palette.getColors(design=dat.design, groups=levels)

    # Transpose data
    dat.trans = dat.transpose()

    dat.trans = dat.trans.loc[:, sample_ids]

    # Run PLS
    df_scores, df_weights, df_classification = runPLS(
        dat.trans, dat.group, args.toCompare, args.nComp, args.cross_validation
    )

    # Update palette afterdrop selection of groups toCompare
    palette.design = palette.design.T[df_scores.index].T
    palette.ugColors = {
        ugc: palette.ugColors[ugc]
        for ugc in list(palette.ugColors.keys())
        if ugc in args.toCompare
    }

    # Plotting scatter plot for scores
    with PdfPages(args.figure) as pdfOut:
        logger.info("Plotting PLS scores")
        plotScores(data=df_scores, palette=palette, pdf=pdfOut)

    # Save df_scores, df_weights and df_classification to tsv files.
    df_scores.to_csv(args.outScores, sep="\t", index_label="sampleID")
    df_weights.to_csv(args.outWeights, sep="\t", index_label=dat.uniqID)
    df_classification.to_csv(args.outClassification, sep="\t", index_label="sampleID")

    # Computing mismatches between original data and final data.
    classification_mismatch_percent = (
        100
        * sum(
            df_classification["Group_Observed"]
            == df_classification["Group_Predicted_Rounded"]
        )
        / df_classification.shape[0]
    )
    classification_mismatch_percent_string = (
        str(classification_mismatch_percent) + " Percent"
    )
    os.system(
        "echo %s > %s"
        % (classification_mismatch_percent_string, args.outClassificationAccuracy)
    )
    logger.info("Finishing PLS execution")


if __name__ == "__main__":
    args = getOptions()
    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel="debug")
    else:
        sl.setLogger(logger)
    logger.info(
        """Importing data with following parameters:
                Input: {0}
                Design: {1}
                uniqID: {2}
                group: {3}
                """.format(
            args.input, args.design, args.uniqID, args.group
        )
    )
    palette = colorHandler(pal=args.palette, col=args.color)
    logger.info(
        "Using {0} color scheme from {1} palette".format(args.color, args.palette)
    )
    main(args)
