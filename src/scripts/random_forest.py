#!/usr/bin/env python
################################################################################
# AUTHOR: Miguel A. Ibarra <miguelib@ufl.edu>
# DESCRIPTION: Take a a wide format file and perform a random forest analysis.
################################################################################

import os
import logging
import argparse
import warnings
from argparse import RawDescriptionHelpFormatter
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from matplotlib.backends.backend_pdf import PdfPages
from secimtools.dataManager import logger as sl
from secimtools.dataManager.interface import wideToDesign
from secimtools.visualManager.module_bar import quickHBar
from secimtools.visualManager.manager_color import colorHandler
from secimtools.visualManager.manager_figure import figureHandler


def getOptions(myopts=None):
    """ Function to pull in arguments """
    description = """ Random Forest """
    parser = argparse.ArgumentParser(description=description,
                                    formatter_class=RawDescriptionHelpFormatter)
    # Standard Input
    standard = parser.add_argument_group(title='Standard input',
                        description='Standard input for SECIM tools.')
    standard.add_argument("-i", "--input", dest="input", action='store',
                        required=True, help="Input dataset in wide format.")
    standard.add_argument("-d", "--design", dest="design", action='store',
                        required=True, help="Design file.")
    standard.add_argument("-id", "--ID", dest="uniqID", action='store',
                        required=True,  help="Name of the column with unique"\
                        " identifiers on wide file.")
    standard.add_argument("-g", "--group", dest="group", action='store',
                        required=True, help="Group/treatment identifier in "\
                        "design file.")
    standard.add_argument("-l","--levels",dest="levels",action="store",
                        required=False, default=False, help="Different groups to"\
                        " sort by separeted by commas.")
    # Tool Input
    tool = parser.add_argument_group(title='Tool specific input',
                            description='Optional/Specific input for the tool.')
    tool.add_argument("-s","--snum", dest="snum", action='store', type=int,
                        required=False,default=1000,help="Number of estimators.")
    tool.add_argument("-n","--num", dest="num", action='store', type=int,
                        required=False,default=20,help="Number of varibles to"\
                        "plot ont Variable Importance Plot")
    # Tool Output
    output = parser.add_argument_group(title='Required output')
    output.add_argument("-o","--out", dest="oname", action='store',
                        required=True, help="Output file name.")
    output.add_argument("-o2","--out2", dest="oname2", action='store',
                        required=True, help="Output file name.")
    output.add_argument("-f","--figure",dest="figure",action="store",
                        required=False,help="Name of output file to store "\
                        "feature importance plots for the model")
    # Plot Options
    plot = parser.add_argument_group(title='Plot options')
    plot.add_argument("-pal","--palette",dest="palette",action='store',required=False,
                        default="sequential", help="Name of the palette to use.")
    plot.add_argument("-col","--color",dest="color",action="store",required=False,
                        default="Blues_9", help="Name of a valid color"
                        " scheme on the selected palette")
    args = parser.parse_args()

    # Standardize paths
    args.input  = os.path.abspath(args.input)
    args.oname  = os.path.abspath(args.oname)
    args.design = os.path.abspath(args.design)
    args.oname2 = os.path.abspath(args.oname2)
    args.figure = os.path.abspath(args.figure)

    if args.levels:
        args.levels = args.levels.split(",")
    return(args)


def plotVarImportance(palette, data, pdf, var):
    """
    Runs LDA over a wide formated dataset

    :Arguments:
        :type scores: pandas.DataFrame
        :param scores: Scores of the LDA.

        :type pdf: pdf object
        :param pdf: PDF object to save all the generated figures.

        :type var: int
        :param var: Number of variables to plot.

    :Returns:
        :rtype scores_df: pandas.DataFrame
        :return scores_df: Scores of the LDA.
    """
    # Subset data upToTheNumberOf Features
    data=data[:var]

    # Sort data
    data=data.sort_values(by="ranked_importance", ascending=True, axis=0)

    # Creating a figure handler instance
    fh = figureHandler(proj='2d', figsize=(8,8))

    # Chomp palette
    palette.chompColors(start=3,end=palette.number)

    # Get color list
    colors = palette.getColorsCmapPalette(data["ranked_importance"])

    # Multiply by 100 to get percentages instead of proportions
    data["ranked_importance"] = data["ranked_importance"]*100

    # Creating plot
    quickHBar(ax=fh.ax[0], values=data["ranked_importance"],
                xticks=data["feature"], colors=colors, lw=0)

    # Formatting axis
    fh.formatAxis(figTitle="Variable Importance Plot", xTitle="%", grid=False,
                yTitle="Features")

    # Adding figure to pdf
    fh.addToPdf(dpi=600,pdfPages=pdf)


def runRFC(data, group, revertStr, origStr, nStim):
    """
    Runs Random Forest Classifier and outputs tables for further export or use.

    :Arguments:
        :type dat: interface.wideToDesign
        :param dat: wideToDesign instance providing acces to all input data.

        :type nStim: int
        :param nStim: Number Stimations.

    :Returns:
        :rtype scores_df: pandas.DataFrame
        :return scores_df: Scores of the LDA.
    """

    # Drope nans from data.
    data.dropna(axis=1, inplace=True)

    # Pull classifications out of dataset.
    classes = data[group].copy()

    # Remove class column from data.
    data.drop(group, axis=1, inplace=True)

    # Build Random Forest classifier
    rfc_model = RandomForestClassifier(n_estimators=nStim)
    rfc_model.fit(data, classes)

    # Identify features and creating a dataFrame for it
    df_importance = pd.DataFrame([data.columns, rfc_model.feature_importances_],
                    index=['feature', 'ranked_importance']).T

    # Sort the dataFrame by importance
    df_importance = df_importance.sort_values(by="ranked_importance", axis=0,
                    ascending=False)

    # Get unfiltered names for importnace (look at interface for more detail)
    df_rev = df_importance.applymap(lambda x: revertStr(x))

    # Select data based on features
    data = data[df_importance["feature"].tolist()]

    # Create a dataframe for the selected data
    reverted_columns = [origStr[x] for x in data.columns]
    data.columns = reverted_columns

    # Convert Series to dataFrame
    df_classes = pd.DataFrame(classes)

    # Reset index on classifications DataFrame
    df_classes.reset_index(inplace=True)

    # Join classifications to the transformed data
    data = df_classes.join(data, on='sampleID')

    return(df_rev, data, df_importance)


def main(args):
    """Perform the Random Forest analysys"""

    # Set the color palette
    palette = colorHandler(pal=args.palette, col=args.color)
    logger.info("Using {0} color scheme from {1} palette".format(args.color, args.palette))

    # Import data through interface
    dat = wideToDesign(args.input, args.design, args.uniqID, args.group,
                        anno=args.levels, clean_string=True, logger=logger)

    # Cleaning from missing data
    dat.dropMissing()
    # Select remaining sample ids for dataframe filtering
    sample_ids = dat.wide.index.tolist()
    # Grab the group column to extract classes from
    sample_ids.append(dat.group)
    data = dat.transpose().loc[:, sample_ids]

    # Run Random Forest Classifier on data.
    logger.info('Creating classifier')
    df_rev, df_transf, df_importance = runRFC(data, dat.group, dat.revertStr, dat.origString, nStim=args.snum)

    # Plot feature importances
    logger.info('Plotting Variable Importance Plot')
    with PdfPages(args.figure) as pdfOut:
        plotVarImportance(palette, data=df_importance, pdf=pdfOut, var=args.num)

    # Exporting Transformed data and df_rev data
    logger.info('Exporting data to TSV format')
    df_transf.to_csv(args.oname, index=False, sep='\t', float_format="%.4f")
    df_rev.to_csv(args.oname2, index=False, sep='\t')


if __name__ == '__main__':
    """Tool is called on the command-line"""
    args = getOptions()
    logger = logging.getLogger()
    sl.setLogger(logger)
    logger.info("Importing data with following parameters: "\
                "\n\tWide: {0}"\
                "\n\tDesign: {1}"\
                "\n\tUnique ID: {2}"\
                "\n\tGroup Column: {3}".format(args.input, args.design,
                args.uniqID, args.group))
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    main(args)

