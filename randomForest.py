#!/usr/bin/env python
#!/usr/bin/env python
################################################################################
# DATE: 2016/Oct/25
#
# SCRIPT: random_forest.py
#
# VERSION: 1.0
# 
# AUTHOR: Miguel A. Ibarra (miguelib@ufl.edu)
# 
# DESCRIPTION: This script takes a a wide format file and performs a # random
#               forest analysis.
#
################################################################################

# Built-in packages
import os
import logging
import argparse
import warnings
from argparse import RawDescriptionHelpFormatter

# Add-on packages
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from matplotlib.backends.backend_pdf import PdfPages

# Local Packages
import logger as sl
from module_bar import quickHBar
from interface import wideToDesign
from manager_color import colorHandler
from manager_figure import figureHandler


def getOptions(myopts=None):
    """ Function to pull in arguments """
    description = """ Random Forest """
    parser = argparse.ArgumentParser(description=description, 
                                    formatter_class=RawDescriptionHelpFormatter)

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

    optional = parser.add_argument_group(title='Tool specific input', 
                            description='Optional/Specific input for the tool.')
    optional.add_argument("-s","--snum", dest="snum", action='store', type=int, 
                        required=False,default=1000,help="Number of estimators.")
    optional.add_argument("-n","--num", dest="num", action='store', type=int, 
                        required=False,default=20,help="Number of varibles to"\
                        "plot ont Variable Importance Plot")

    output = parser.add_argument_group(title='Required output')
    output.add_argument("-o","--out", dest="oname", action='store', 
                        required=True, help="Output file name.")
    output.add_argument("-o2","--out2", dest="oname2", action='store', 
                        required=True, help="Output file name.")
    output.add_argument("-f","--figure",dest="figure",action="store",
                        required=False,help="Name of output file to store "\
                        "feature importance plots for the model")

    plot = parser.add_argument_group(title='Plot options')
    plot.add_argument("-pal","--palette",dest="palette",action='store',required=False, 
                        default="sequential", help="Name of the palette to use.")
    plot.add_argument("-col","--color",dest="color",action="store",required=False, 
                        default="Blues_9", help="Name of a valid color"
                        " scheme on the selected palette")

    args = parser.parse_args()
    return(args)

def plotVarImportance(features, pdf, var):
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
    features=features[:var]

    # Sort data
    features=features.sort_values(by="ranked_importance", ascending=True, axis=0)                        

    # Creating a figure handler instance
    fh = figureHandler(proj='2d', figsize=(8,8))

    # Chomp palette
    palette.chompColors(start=3,end=palette.number)

    # Get color list
    colors = palette.getColorsCmapPalette(features["ranked_importance"])

    # Multiply by 100 to get percentages instead of proportions
    features["ranked_importance"] = features["ranked_importance"]*100

    # Creating plot
    quickHBar(ax=fh.ax[0], values=features["ranked_importance"],
                xticks=features["feature"], colors=colors, lw=0)

    # Formatting axis
    fh.formatAxis(figTitle="Variable Importance Plot", xTitle="%", grid=False,
                yTitle="Features")


    # Adding figure to pdf
    fh.addToPdf(dpi=600,pdfPages=pdf)

def runRFC(dat,nStim):
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

    # Transpose data.
    data = dat.transpose()

    # Drope nans from data.
    data.dropna(axis=1, inplace=True)

    # Pull classifications out of dataset.
    classes = data[dat.group].copy()

    # Remove class column from data.
    data.drop(dat.group, axis=1, inplace=True)

    # Build Random Forest classifier
    rfc_model = RandomForestClassifier(n_estimators=nStim)
    rfc_model.fit(data, classes)

    # Identify features and creating a dataFrame for it
    importance_df = pd.DataFrame([data.columns, rfc_model.feature_importances_], 
                    index=['feature', 'ranked_importance']).T

    # Sort the dataFrame by importance
    importance_df = importance_df.sort_values(by="ranked_importance", axis=0, 
                    ascending=False)

    # Get no filtered names for importnace (look at interface for more detail)
    rev_df = importance_df.applymap(lambda x: dat.revertStr(x))

    # Select data based on features
    data = data[importance_df["feature"].tolist()]

    # Create a dataframe for the selected data
    transf_df = pd.DataFrame(rfc_model.transform(data, threshold=0))
    transf_df.columns = [dat.revertStr(x) for x in data.columns]

    # Convert Series to dataFrame
    classes = pd.DataFrame(classes)

    # Reset index on classifications DataFrame
    classes.reset_index(inplace=True)

    # Join classifications to the transformed data
    transf_df = classes.join(transf_df)

    return(rev_df,transf_df,importance_df)

def main(args):
    # Import data through interface
    dat = wideToDesign(args.input, args.design, args.uniqID, args.group, 
                        clean_string=True)

    #Run Random Forest Classifier on data.
    logger.info('Creating classifier')
    rev_df,transf_df,importance_df = runRFC(dat, nStim=args.snum)

    # Plot feature importances
    logger.info('Plotting Variable Importance Plot')
    with PdfPages(os.path.abspath(args.figure)) as pdfOut:
        plotVarImportance(features=importance_df, pdf=pdfOut, var=args.num)
    
    # Exporting Transformed data
    logger.info('Exporting transformed data')
    transf_df.to_csv(args.oname, index=False, sep='\t', float_format="%.4f")

    # Selectim columns for rev_df data frame
    logger.info('Exporting features')
    rev_df.to_csv(args.oname2, index=False, sep='\t')

if __name__ == '__main__':
    # Command line options
    args = getOptions()

    # Activate Logger
    logger = logging.getLogger()
    sl.setLogger(logger)

    # Starting script
    logger.info(u"Importing data with following parameters: "\
                "\n\tWide: {0}"\
                "\n\tDesign: {1}"\
                "\n\tUnique ID: {2}"\
                "\n\tGroup Column: {3}".format(args.input, args.design, 
                args.uniqID, args.group))

    # Stablishing color palette
    palette = colorHandler(pal=args.palette, col=args.color)
    logger.info(u"Using {0} color scheme from {1} palette".format(args.color,
                args.palette))

    # Getting rid of warnings.
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Main Code
    main(args)

