#!/usr/bin/env python
######################################################################################
# DATE: 2017/06/19
#
# MODULE: ttest_single_group.py
#
# VERSION: 1.0
#
# AUTHOR: Alexander Kirpich <akirpich@ufl.edu>
#
# DESCRIPTION: This tool runs t-test which can be either single, sample, or differences
#
#######################################################################################

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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Importing cross-validation functions
from sklearn import datasets
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
# Importing single sample t-test module
from scipy.stats import ttest_1samp
# Import local plotting libraries
from secimtools.visualManager import module_box as box
from secimtools.visualManager import module_hist as hist
from secimtools.visualManager import module_lines as lines
from secimtools.visualManager import module_scatter as scatter
from secimtools.visualManager.manager_color import colorHandler
from secimtools.visualManager.manager_figure import figureHandler
# Import local data libraries
from secimtools.dataManager import logger as sl
from secimtools.dataManager.interface import wideToDesign

def getOptions(myopts=None):
    """ Function to pull in arguments """
    description="""
    This script runs a t-test for a single sample for each feature in the data.
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
    standard.add_argument("-id", "--uniqueID",dest="uniqueID", action='store', required=True,
                        help="Name of the column with unique identifiers.")
    standard.add_argument("-g", "--group",dest="group", action='store', required=False,
                        default=False, help="Name of the column with group variable.")
    standard.add_argument("-mu", "--mu",dest="mu", action='store', required=False,
                        default=0, help="Mu value for the null.")
    # Tool output
    output = parser.add_argument_group(title='Required output')
    output.add_argument("-s","--summaries",dest="summaries",action='store',required=True,
                        help="Summaries file name. TSV format.")
    output.add_argument("-f","--flags",dest="flags",action='store',required=True,
                        help="Flags file. TSV format.")
    output.add_argument("-v","--volcano",dest="volcano",action="store",required=True,
                        help="Volcano plot. PDF Format.")
    # Plot Options
    plot = parser.add_argument_group(title='Plot options')
    plot.add_argument("-pal","--palette",dest="palette",action='store',required=False,
                        default="tableau", help="Name of the palette to use.")
    plot.add_argument("-col","--color",dest="color",action="store",required=False,
                        default="Tableau_20", help="Name of a valid color scheme"\
                        " on the selected palette")
    args = parser.parse_args()

    # Standardized output paths
    args.input                     = os.path.abspath(args.input)
    args.design                    = os.path.abspath(args.design)
    args.summaries                 = os.path.abspath(args.summaries)
    args.flags                     = os.path.abspath(args.flags)
    args.volcano                   = os.path.abspath(args.volcano)

    return(args)

def main(args):
	if args.group == False:
		logger.info(u"""t-test will be performed for the entire dataset since goruping variable was not provided.""")

        # Loading data trough the interface
		logger.info("Loading data with the Interface")
		dat = wideToDesign(args.input, args.design, args.uniqueID, logger=logger)

        # Treat everything as numeric
		dat.wide = dat.wide.applymap(float)

        # Cleaning from missing data
		dat.dropMissing()

        # Saving the number of unique groups that will be used for plotting.
        # Since we did not feed any grouping variable it is exactly one.
		number_of_unique_groups = 1

        # Extracting data from the interface.
		data_frame = dat.wide.transpose()
        # Extracting number of features. We do not subtract 1 since we have not provided args.group
		number_of_features = data_frame.shape[1]
        # Saving treatment group name from the arguments.

        # Computing overall summaries (mean and variance).
        # This part just produces sumamry statistics for the output table.
        # This has nothing to do with single sample t-test. This is just summary for the table.
		mean_value_all = [0] * number_of_features
		variance_value_all = [0] * number_of_features
        # Creating array of means for the current group that will be filled.
        # Creating p_values, neg_log10_p_value, flag_values, difference_value lists filled wiht 0es.
		p_value           = [0] * number_of_features
		t_value           = [0] * number_of_features
		neg_log10_p_value = [0] * number_of_features
		difference_value  = [0] * number_of_features
		flag_value_0p01   = [0] * number_of_features
		flag_value_0p05   = [0] * number_of_features
		flag_value_0p10   = [0] * number_of_features

		for j in range(0, number_of_features ):
            # We transpose here so data will be easier to operate on.
			data_frame_manipulate_transpose  = data_frame.transpose()
            # Pulling indexes list from the current data frame.
			indexes_list_complete = data_frame_manipulate_transpose.index.tolist()

            # Computing dataset summaries.
			mean_value_all[j] = np.mean(data_frame_manipulate_transpose.loc[ indexes_list_complete[j] ]) 
			variance_value_all[j] = np.var(data_frame_manipulate_transpose.loc[ indexes_list_complete[j] ], ddof = 1)

			ttest_1samp_args = [ data_frame_manipulate_transpose.loc[ indexes_list_complete[j] ] , float(args.mu) ]
			logger.info(ttest_1samp_args)
			p_value[j] = ttest_1samp( *ttest_1samp_args )[1]
			#logger.info(p_value[j])

        # Creating the table and putting the results there.
		summary_df     =  pd.DataFrame(data = mean_value_all, columns = ["GrandMean"], index = indexes_list_complete )
		summary_df['SampleVariance']              =  variance_value_all
		summary_df['p_value']           = p_value


        # Rounding the results up to 4 precision digits.
		summary_df = summary_df.apply(lambda x: x.round(4))

    	# Adding name for the unique ID column that was there oroginally.
		summary_df.index.name    =  args.uniqueID

    	# Save summary_df to the ouptut
		summary_df.to_csv(args.summaries, sep="\t")

if __name__ == '__main__':
	args = getOptions()
	logger = logging.getLogger()
	sl.setLogger(logger)
	logger.info(u"""Importing data with following parameters:
                Input: {0}
                Design: {1}
                UniqueID: {2}
                Group: {3}
                """.format(args.input, args.design, args.uniqueID, args.group ))
	palette = colorHandler(pal=args.palette, col=args.color)
	logger.info(u"Using {0} color scheme from {1} palette".format(args.color, args.palette))
	warnings.filterwarnings("ignore")
	main(args)
