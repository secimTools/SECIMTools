#!/usr/bin/env python
################################################################################
# DATE: 2017/06/29
#
# SCRIPT: linear_discriminant_analysis.py
#
# VERSION: 2.0
#
# AUTHORS: Miguel A. Ibarra <miguelib@ufl.edu> and Alexander Kirpich <akirpich@ufl.edu>
#
# DESCRIPTION: This script takes a a wide format file and performs a linear
# discriminant analysis with/without single or double cross-validation.
#
################################################################################
# Import future libraries
from __future__ import division

# Import built-in libraries
import os
import logging
import argparse
import warnings
from itertools import combinations
from argparse import RawDescriptionHelpFormatter

# Import add-on libraries
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Importing cross-validation functions
from sklearn import datasets
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler



# Import local plottin libraries
from secimtools.visualManager import module_scatter as scatter
from secimtools.visualManager.manager_color import colorHandler
from secimtools.visualManager.manager_figure import figureHandler

# Import local data libraries
from secimtools.dataManager import logger as sl
from secimtools.dataManager.interface import wideToDesign

def getOptions(myopts=None):
    """ Function to pull in arguments """
    description="""
    This script runs a Linear Discriminant Analysis (LDA)
    """
    # Standard Input
    parser = argparse.ArgumentParser(description=description,
                                    formatter_class=RawDescriptionHelpFormatter)
    standard = parser.add_argument_group(title='Standard input',
                                description='Standard input for SECIM tools.')
    standard.add_argument( "-i","--input", dest="input", action='store',
                        required=True, help="Input dataset in wide format.")
    standard.add_argument("-d" ,"--design",dest="design", action='store',
                        required=True, help="Design file.")
    standard.add_argument("-id", "--ID",dest="uniqID", action='store', required=True,
                        help="Name of the column with unique identifiers.")
    standard.add_argument("-g", "--group",dest="group", action='store', required=True,
                        default=False,help="Name of the column with groups.")
    standard.add_argument("-l","--levels",dest="levels",action="store",
                        required=False, default=False, help="Different groups to"\
                        " sort by separeted by commas.")
    # Tool Input
    tool = parser.add_argument_group(title='Tool input',
                            description='Optional/Specific input for the tool.')
    tool.add_argument('-cv', "--cross_validation", dest="cross_validation", action='store',
                        required=True, help="Choice of cross-validation procedure for the -nc determinantion: none, "\
                        "single, double.")
    tool.add_argument("-nc", "--nComponents",dest="nComponents", action='store',
                        type= int, required=False, default=None,
                        help="Number of components [Default == 2]. Used only if -cv=none.")
    # Tool output
    output = parser.add_argument_group(title='Required output')
    output.add_argument("-o","--out",dest="out",action='store',required=True,
                        help="Name of output file to store scores. TSV format.")
    output.add_argument("-oc","--outClassification",dest="outClassification",action='store',required=True,
                        help="Name of output file to store classification. TSV format.")
    output.add_argument("-oca","--outClassificationAccuracy",dest="outClassificationAccuracy",action='store',required=True,
                        help="Name of output file to store classification accuracy. TSV format.")
    output.add_argument("-f","--figure",dest="figure",action="store",required=True,
                        help="Name of output file to store scatter plots for scores")
    # Plot Options
    plot = parser.add_argument_group(title='Plot options')
    plot.add_argument("-pal","--palette",dest="palette",action='store',required=False,
                        default="tableau", help="Name of the palette to use.")
    plot.add_argument("-col","--color",dest="color",action="store",required=False,
                        default="Tableau_20", help="Name of a valid color scheme"\
                        " on the selected palette")
    args = parser.parse_args()

    # Standardized output paths
    args.out                       = os.path.abspath(args.out)
    args.input                     = os.path.abspath(args.input)
    args.design                    = os.path.abspath(args.design)
    args.figure                    = os.path.abspath(args.figure)
    args.outClassification         = os.path.abspath(args.outClassification)
    args.outClassificationAccuracy = os.path.abspath(args.outClassificationAccuracy)


    # Split levels if levels
    if args.levels:
        args.levels = args.levels.split(",")

    return(args)

def runLDA(dat,nComp,cv_status):
    """
    Runs LDA over a wide formated dataset

    :Arguments:
        :type dat: interface.wideToDesign
        :param dat: wideToDesign instance providing acces to all input data.

        :type nComp: int
        :param nComp: Number of components to use, if None the program will
                        calculate n_groups -1. If cv_status is single or double this parameter is ingored.

        :type cv_status: string
        :param cv_status: Cross-validation status for nComp. Can be none, single and double.
              If "none" no cross-validation is performed and nComp specified by the user is used instead.

    :Returns:
        :rtype scores_df: pandas.DataFrame
        :return scores_df: Scores of the LDA.
    """


    # The code below depends on cross validation status. The status can be either "single", "double" or "none".


    # Case 1: User provides cv_status = "none". No cross-validation will be performed.
    # The number of components shoul be specified by the user in the nComp variable. The default in xml shoul be 2.
    if cv_status == "none":

       # Telling the user that we are using the number of components pre-specified by the user.
       logger.info(u"Using the number of components pe-specified by the user.")

       # If by mistake the number the componentes nComp was not provided we hardcode it to 2.
       if nComp is None:
          logger.info(u"The number of componets was not provided! Default number 2 is used.")
          nComp = 2

       # Putting the user defined (or corrected if nothing imputted) number of compoents nComp directly into index_min.
       index_min = nComp



    # Case 2: User provides cv_status = "single". Only single cross-validation will be performed.
    if cv_status == "single":

       # Telling the user that we are using the number of components determined via a single cross-validation.
       logger.info(u"Using the number of components determined via a single cross-validation.")

       # Pulling the number of unique groups that we will feed to cross-validation.
       group_values_series = dat.transpose()[dat.group].T.squeeze()

       # Checking if the sample sizes is smaller than 100 and exiting if that is the case.
       if (len(group_values_series) < 100):
          logger.info(u"The required number of samples for a single cross-validation procedure is at least 100. The dataset has {0}.".format(len(group_values_series)))
          logger.info(u"Exiting the tool.")
          exit()

       group_values_series_unique = group_values_series.unique()
       number_of_unique_groups = group_values_series_unique.shape[0]
       # Ensuring that we will produce at least single plot.
       n_max = max( number_of_unique_groups - 1, 2 )


       # Creating a list of values to perform single cross-validation over.
       # P.S. We do not consider scenario of a single component so that we can produce at least single 2D plot in the end.
       n_list = range(2, n_max + 1)


       # Creating dictionary we gonna feed to the single cross-validation procedure.
       n_list_dictionary = dict( n_components = n_list )


       # Creating a gridsearch object with parameter "n_list_dictionary"
       internal_cv = GridSearchCV( estimator = LinearDiscriminantAnalysis(), param_grid = n_list_dictionary )


       # Performing internal_cv.
       internal_cv.fit( dat.wide.T, dat.transpose()[dat.group] )
       # Assigning index_min from the best internal_cv i.e. internal_cv.best_params_['n_components']
       index_min = internal_cv.best_params_['n_components']



    # Case 3: User provides cv_status = "double". Double cross-validation will be performed.
    if cv_status == "double":

       # Telling the user that we are using the number of components determined via a double cross-validation.
       logger.info(u"Using the number of components determined via a double cross-validation.")

       # Pulling the number of unique groups that we will feed to cross-validation.
       group_values_series = dat.transpose()[dat.group].T.squeeze()

       # Checking if the sample sizes is smaller than 100 and exiting if that is the case.
       if (len(group_values_series) < 100):
          logger.info(u"The required number of samples for a double cross-validation procedure is at least 100. The dataset has {0}.".format(len(group_values_series)))
          logger.info(u"Exiting the tool.")
          exit()

       group_values_series_unique = group_values_series.unique()
       number_of_unique_groups = group_values_series_unique.shape[0]
       # Ensuring that we will produce at least single plot.
       n_max = max( (number_of_unique_groups - 1), 2 )


       # Here we are looping over possible lists we have to cross-validate over.
       # We do not consider scenario of a single components so that we can produce at least one 2D plot in the end.

       # Creating index of the minimum variable that will give us the best prediction.
       # This will be updated during interlan and external CV steps if necessary.
       index_min = 2

       for n_current in range(2, n_max+1):

       # Creating the set of candidates that we will use for both cross-validation loops: internal and external
       # n_list = range(2, n_current + 1)
           n_list = range(2, n_current+1)

           # Creating dictionary we gonna feed to the internal cross-validation procedure.
           n_list_dictionary = dict( n_components = n_list )

       # Creating a gridsearch object with parameter "n_list_dictionary"
           internal_cv = GridSearchCV( estimator = LinearDiscriminantAnalysis(), param_grid = n_list_dictionary)

           # Performing internal_cv for debugging purposes.
           internal_cv.fit(dat.wide.T, dat.transpose()[dat.group])

           # Performing external_cv using internal_cv
           external_cv = cross_val_score(internal_cv, dat.wide.T, dat.transpose()[dat.group])


           # Checking whether adding this extra component to our anlaysis will help.
       # For the first 2 components we assume they are the best and update it later if necessary.
           if n_current == 2:
              best_predction_proportion = external_cv.mean()

           else:
              # Checking whether adding this extra component helped to what we already had.
              if external_cv.mean() > best_predction_proportion:
                   best_predction_proportion = external_cv.mean()
                   index_min = n_current




    #Initialize LDA class and stablished number of coponents
    lda = LinearDiscriminantAnalysis( n_components=index_min )


    # Computing scores for the fit first and savin them into a data frame.

    # Get scores of LDA (Fit LDA)
    scores = lda.fit_transform( dat.wide.T,dat.transpose()[dat.group] )

    # Create column names for scores file
    LVNames = ["Component_{0}".format(i+1) for i in range(index_min)]

    # Create pandas dataframe for scores
    scores_df = pd.DataFrame(scores,columns=LVNames,index=dat.wide.columns.values)

    # Dealing with predicted and classification data frame.

    # Geting predicted values of LDA (Fit LDA)
    fitted_values = lda.predict( dat.wide.T )
    # Pulling the groups from the orogonal data frmae and save it to original_values.
    original_values = dat.transpose()[dat.group].T.squeeze()

    # Combining results into the data_frame so that it can be exported.
    classification_df = pd.DataFrame( {'Group_Observed': original_values ,
                           'Group_Predicted': fitted_values } )

    # Return scores for LDA
    return scores_df, classification_df



def plotScores(data, palette, pdf):
    """
    Runs LDA over a wide formated dataset

    :Arguments:
        :type data: pandas.DataFrame
        :param data: Scores of the LDA.

        :type palette: colorManager.object
        :param palette: Object from color manager

        :type pdf: pdf object
        :param pdf: PDF object to save all the generated figures.

    :Returns:
        :rtype scores_df: pandas.DataFrame
        :return scores_df: Scores of the LDA.
    """
    # Create a scatter plot for each combination of the scores
    for x, y in list(combinations(data.columns.tolist(),2)):

        # Create a single-figure figure handler object
        fh = figureHandler(proj="2d", figsize=(14,8))

        # Create a title for the figure
        title = "{0} vs {1}".format(x,y)

        # Plot the scatterplot based on data
        scatter.scatter2D(x=list(data[x]), y=list(data[y]),
                         colorList=palette.design.colors.tolist(), ax=fh.ax[0])

        # Create legend
        fh.makeLegend(ax=fh.ax[0],ucGroups=palette.ugColors,group=palette.combName)

        # Shrink axis to fit legend
        fh.shrink()

        # Despine axis
        fh.despine(fh.ax[0])

        # Formatting axis
        fh.formatAxis(figTitle=title,xTitle="Scores on {0}".format(x),
            yTitle="Scores on {0}".format(y),grid=False)

        # Adding figure to pdf
        fh.addToPdf(dpi=600,pdfPages=pdf)

def main(args):
    # Checking if levels
    if args.levels and args.group:
        levels = [args.group]+args.levels
    elif args.group and not args.levels:
        levels = [args.group]
    else:
        levels = []

    #Loading data trought Interface
    dat = wideToDesign(args.input, args.design, args.uniqID,group=args.group,
            anno=args.levels, logger=logger)

    # Treat everything as numeric
    dat.wide = dat.wide.applymap(float)

    # Cleaning from missing data
    dat.dropMissing()

    # Get colors for each sample based on the group
    palette.getColors(design=dat.design, groups=levels)

    #Run LDA
    logger.info(u"Runing LDA on data")
    df_scores, df_classification  = runLDA(dat, nComp=args.nComponents, cv_status=args.cross_validation)

    # Plotting scatter plot for scores
    logger.info(u"Plotting LDA scores")
    with PdfPages(args.figure) as pdfOut:
        plotScores(data=df_scores, palette=palette, pdf=pdfOut)

    # Save scores
    df_scores.to_csv(args.out, sep="\t", index_label="sampleID")
    # Save classification
    df_classification.to_csv(args.outClassification, sep="\t", index_label="sampleID")

    # Computing mismatches between original data and final data.
    classification_mismatch_percent = 100 * sum( df_classification['Group_Observed'] == df_classification['Group_Predicted'] )/df_classification.shape[0]
    classification_mismatch_percent_string = str( classification_mismatch_percent ) + ' Percent'
    os.system("echo %s > %s"%( classification_mismatch_percent_string, args.outClassificationAccuracy ) )


    #Ending script
    logger.info(u"Finishing running of LDA")

if __name__ == '__main__':
    # Command line options
    args = getOptions()

    #Set logger
    logger = logging.getLogger()
    sl.setLogger(logger)

    #Starting script
    logger.info(u"""Importing data with following parameters:
                Input: {0}
                Design: {1}
                uniqID: {2}
                group: {3}
                """.format(args.input, args.design, args.uniqID, args.group))

    # Stablishing color palette
    palette = colorHandler(pal=args.palette, col=args.color)
    logger.info(u"Using {0} color scheme from {1} palette".format(args.color,
                args.palette))

    # Getting rid of warnings.
    warnings.filterwarnings("ignore")

    # Main code
    main(args)
