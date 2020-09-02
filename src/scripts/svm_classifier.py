#!/usr/bin/env python
################################################################################
# DATE: 2017/06/29
#
# SCRIPT: svm_classifier.py
#
# VERSION: 2.0
# 
# AUTHORS: Coded by: Ali Ashrafi (a.ali.ashrafi@gmail.com>),
#		     Miguel A Ibarra (miguelib@ufl.edu, 
#                    and Alexander Kirpich (akirpich@ufl.edu)
# 
# DESCRIPTION: This script flags features based on a given threshold.
#
################################################################################
# Import built-in libraries
import os, sys
import logging
import argparse
from argparse import RawDescriptionHelpFormatter

# Import add-on libraries
import numpy as np
import pandas as pd
from pandas import DataFrame as DF
from pandas import read_csv, read_table
from sklearn import svm
# Importing cross-validation functions
from sklearn import datasets
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Import local libraries
from secimtools.dataManager import logger as sl
from secimtools.dataManager.interface import wideToDesign

def getOptions(myOpts=None):
    parser = argparse.ArgumentParser( formatter_class=RawDescriptionHelpFormatter )
    # Standard Input
    standard = parser.add_argument_group(description="Standard Input")
    standard.add_argument('-trw',"--train_wide", dest="train_wide", action='store',
                        required=True, help="wide part of the train dataset.")
    standard.add_argument('-trd',"--train_design", dest="train_design", 
                        action='store', required=True, help="design part of "\
                        "the train dataset.")
    standard.add_argument('-tew',"--test_wide", dest="test_wide", action='store',
                        required=True, help="wide part of the test dataset.")
    standard.add_argument('-ted',"--test_design", dest="test_design", 
                        action='store', required=True, help="design part of"\
                        " the test dataset.")
    standard.add_argument('-g',"--group", dest="group",action='store',
                        required=True, default=False, help="Name of column in design file"\
                        " with Group/treatment information.")
    standard.add_argument('-id',"--ID", dest="uniqID", action='store',
                        required=True, help="Name of the column with unique "\
                        "identifiers.")
    # Tool Input
    tool = parser.add_argument_group(description="Tool Input")
    tool.add_argument('-k',"--kernel", dest="kernel", action='store',
                        required=True, help="choice of kernel function: rbf, "\
                        "linear, poly, sigmoid.")
    tool.add_argument('-d',"--degree", dest="degree", action='store',
                        required=True, help="(integer) degree for the polynomial"\
                        " kernel, default 3.")
    tool.add_argument('-c',"--C", dest="C", action='store', required=True, 
                        help="positive regularization parameter. This parameter is ignored when -cv is single or double")
    tool.add_argument('-cv', "--cross_validation", dest="cross_validation", action='store',
                        required=True, help="Choice of cross-validation procedure for the regularization parameter -c determinantion: none, "\
                        "single, double.")
    tool.add_argument('-c_lower_bound',"--C_lower_bound", dest="C_lower_bound", action='store', required=False, 
                        help="positive regularization parameter lower bound. Ignored if -cv is none and -c is specified.")
    tool.add_argument('-c_upper_bound',"--C_upper_bound", dest="C_upper_bound", action='store', required=False, 
                        help="positive regularization parameter upper bound. Ignored if -cv is none and -c is specified.")
    tool.add_argument('-a',"--a", dest="a", action='store', required=True, 
                        help=" positive coefficient in kernel function.")
    tool.add_argument('-b',"--b", dest="b", action='store', required=True, 
                        help=" independent term coefficient in kernel function.")
    # Tool Output 
    output = parser.add_argument_group(description="Output Paths")
    output.add_argument("-oc","--outClassification",dest="outClassification", action='store', required=True,
                        help="Name of the output file to store classification performed on the traing data set. TSV format.")
    output.add_argument('-oca',"--outClassificationAccuracy", dest="outClassificationAccuracy", action='store',
                         required=True, help="Output classification accuracy value on the training data set.")
    output.add_argument("-op","--outPrediction",dest="outPrediction", action='store', required=True,
                        help="Name of the output file to store prediction performed on the target data set. TSV format.")
    output.add_argument('-opa',"--outPredictionAccuracy", dest="outPredictionAccuracy", action='store',
                         required=True, help="Output prediction accuracy value on the target data set.")
    args = parser.parse_args()

    # Standardize paths
    args.test_wide                  = os.path.abspath(args.test_wide)
    args.train_wide                 = os.path.abspath(args.train_wide)
    args.test_design                = os.path.abspath(args.test_design)
    args.train_design               = os.path.abspath(args.train_design)
    args.outClassification          = os.path.abspath(args.outClassification)
    args.outClassificationAccuracy  = os.path.abspath(args.outClassificationAccuracy)
    args.outPrediction              = os.path.abspath(args.outPrediction)
    args.outPredictionAccuracy      = os.path.abspath(args.outPredictionAccuracy)

    return(args)

def correctness(x):
    if x[args.group]==x['predicted_class']:
        return 1
    else:
        return 0

def getAccuracy(data):
    data['correct']=data.apply(correctness,axis=1)
    accuracy=float(data['correct'].sum())/data.shape[0]
    return accuracy

def main(args):

    # Load target dataset trought the interface.
    target = wideToDesign(wide=args.test_wide, design = args.test_design, 
                          uniqID=args.uniqID, group=args.group, logger=logger)
     
    # Load training dataset trought the interface.
    train =  wideToDesign(wide=args.train_wide, design= args.train_design, 
                          uniqID=args.uniqID, group=args.group, logger=logger)
    
    # Treating everything as numeric.
    train.wide = train.wide.applymap(float)
    target.wide = train.wide.applymap(float)

    # Dropping missing values
    train.dropMissing()
    train = train.transpose()

    # Dropping missing values
    target.dropMissing()
    target = target.transpose()

    # Making sure test and train have the same features
    for i in target.columns:
        if i not in train.columns:
            del target[i]


    # Saving input parameters into variables that will be easier to manipulate in the future.
    cv_status = args.cross_validation
    kernel_final = args.kernel   
    gamma_final  = float(args.a)
    coef0_final  = float(args.b)
    degree_final = int(args.degree)

    # Definging the data to use for the model training.
    train_classes_to_feed = train[args.group].copy()
    train_data_to_feed = train
    del train_data_to_feed[args.group]
    # Definging the data to use for the model target.
    target_classes_to_feed = target[args.group].copy()
    target_data_to_feed = target
    del target_data_to_feed[args.group]

    # Debugging step
    # print "train_classes_to_feed = ", train_classes_to_feed
    # print "len(train_classes_to_feed) = ", len(train_classes_to_feed)
    # Debugging step
    # print "target_classes_to_feed = ", target_classes_to_feed
    # print "target_data_to_feed = ", target_data_to_feed


    # Debuggin step: we are checking what is the status for cross-validation procedure specified by the user.
    # print "Cross-validation choice: ", cv_status


    # The code below depends on cross validation status. The status can be either "none", "single" or "double".

    # Case 1: User provides cv_status = "none". No cross-validation will be performed.	
    # The value of C has to be specified by the user and pulled from the user's input.
    if cv_status == "none":
 
       # Telling the user that we are using the number of components pre-specified by the user.
       logger.info(u"Using the value of C specified by the user.")

       # Putting the user defined C value into C_final variable.
       C_final = float(args.C)



    # Case 2: User provides cv_status = "single". Only single cross-validation will be performed for the value of C.	
    if cv_status == "single":

       # Telling the user that we are using the C penalty determined via a single cross-validation.
       logger.info(u"Using the value of C determined via a single cross-validation.")

       # Checking if the sample sizes is smaller than 100 and exiting if that is the case.
       if (len(train_classes_to_feed) < 100):
	  logger.info(u"The required number of samples for a single cross-validation procedure is at least 100. The dataset has {0}.".format(len(train_classes_to_feed)))
	  logger.info(u"Exiting the tool.")
          exit()	

       # Defining boundaries of C used for a grid for a cross-validation procedure that user has supplied.
       C_lower = float(args.C_lower_bound)
       C_upper = float(args.C_upper_bound)
       
       # Debugging step.  
       # print "C_lower", C_lower
       # print "C_upper", C_upper
       
       # Creating a list of values to perform a single cross-validation procedure over a grid.	
       # We tell the user that the user-specified range will be splitted into 20 pieces and each value will be investigated in cross-validation procedure.
       C_list_of_values =  np.linspace(C_lower, C_upper, 20)

       # Creating dictionary we gonna feed to the single cross-validation procedure.
       # In this disctionary gamma is speficied by the user. We are only cross-validating over the value of C.
       parameter_list_of_values_dictionary_gamma_specified = { "kernel": [kernel_final],
                                                               "C":      C_list_of_values, 
                                                               "gamma":  [gamma_final],
				                	       "coef0": [coef0_final],
                                                               "degree": [degree_final] }
       # In this disctionary gamma is determined automatically if the first dictionary fails. 
       parameter_list_of_values_dictionary_gamma_auto      = { "kernel": [kernel_final],
                                                               "C":      C_list_of_values, 
                                                               "gamma":  ["auto"],
				                	       "coef0": [coef0_final],
                                                               "degree": [degree_final] }

       # Debugging step
       # Printing the dictionary that has just been created.
       # print "parameter_list_of_values_dictionary_gamma_specified = ", parameter_list_of_values_dictionary_gamma_specified
       # print "parameter_list_of_values_dictionary_gamma_auto      = ", parameter_list_of_values_dictionary_gamma_auto

       # Definging the fit depending on the gamma value.
       try:
           logger.info("Running SVM model")
           # Creating a gridsearch object with parameter "C_list_of_values_dictionary"
           internal_cv = GridSearchCV( estimator = SVC(), param_grid = parameter_list_of_values_dictionary_gamma_specified )

       except:
           logger.info("Model failed with gamma = {0} trying automatic gamma "\
                        "instead of.".format(float(args.a)))
           # Creating a gridsearch object with parameter "C_list_of_values_dictionary"
           internal_cv = GridSearchCV( estimator = SVC(), param_grid = parameter_list_of_values_dictionary_gamma_auto )


       # Performing internal_cv.
       internal_cv.fit(train_data_to_feed, train_classes_to_feed)

       # Debugging step.
       # print "internal_cv.fit(train_data_to_feed, train_classes_to_feed) = ", internal_cv.fit(train_data_to_feed, train_classes_to_feed) 
       # print "train_data_to_feed =", train_data_to_feed
       # print "train_classes_to_feed =", train_classes_to_feed
       # print "internal_cv.best_score_ = ",  internal_cv.best_score_
       # print "internal_cv.cv_results_ = ", internal_cv.cv_results_
       # print "internal_cv.best_score_ = ", internal_cv.best_score_
       # print "internal_cv.best_params_ = ", internal_cv.best_params_['C']
       # print "internal_cv.cv_results_['params'][internal_cv.best_index_] = ", internal_cv.cv_results_['params'][internal_cv.best_index_] 

       # Assigning C_final from the best internal_cv i.e. internal_cv.best_params_['C'] 
       C_final = internal_cv.best_params_['C']



    # Case 3: User provides cv_status = "double". Double cross-validation will be performed.
    if cv_status == "double":

       # Telling the user that we are using the C penalty determined via a double cross-validation.
       logger.info(u"Using the value of C determined via a double cross-validation.")


       # Checking if the sample sizes is smaller than 100 and exiting if that is the case.
       if (len(train_classes_to_feed) < 100):
	  logger.info(u"The required number of samples for a double cross-validation procedure is at least 100. The dataset has {0}.".format(len(train_classes_to_feed)))
	  logger.info(u"Exiting the tool.")
          exit()	

       # Defining boundaries of C used for a grid for a cross-validation procedure that user has supplied.
       C_lower = float(args.C_lower_bound)
       C_upper = float(args.C_upper_bound)
       
       # Debugging step.  
       # print "C_lower", C_lower
       # print "C_upper", C_upper
       
       # Creating a list of values to perform single cross-validation over.	
       # We tell the user that the user-specified range will be splitted into 20 pieces and each value will be investigated in cross-validation procedure.
       C_list_of_values =  np.linspace(C_lower, C_upper, 20)

       # Creating C_final equal to the first element of indexed array C_list_of_values. 
       # This will be updated during internal and external CV steps if necessary.
       C_final = C_list_of_values[0]

       for index_current in range(0, 20):

	   # Creating the set of candidates that we will use for both cross-validation loops: internal and external
	   C_list_of_values_current =  np.linspace(C_list_of_values[0], C_list_of_values[index_current], (index_current+1) )
  
           # Creating dictionary we gonna feed to the single cross-validation procedure.
           # In this disctionary gamma is speficied by the user.
           parameter_list_of_values_dictionary_gamma_specified = { "kernel": [kernel_final],
                                                                   "C":      C_list_of_values_current, 
                                                                   "gamma":  [gamma_final],
		                                      	           "coef0":  [coef0_final],
                                                                   "degree": [degree_final] }
           # In this disctionary gamma is determined automatically if the first dictionary fails. 
           parameter_list_of_values_dictionary_gamma_auto      = { "kernel": [kernel_final],
                                                                   "C":      C_list_of_values_current, 
                                                                   "gamma":  ["auto"],
				                	           "coef0":  [coef0_final],
                                                                   "degree": [degree_final] }

           # Debugging step
           # Printing the dictionary that has just been created.
           # print "parameter_list_of_values_dictionary_gamma_specified = ", parameter_list_of_values_dictionary_gamma_specified
           # print "parameter_list_of_values_dictionary_gamma_auto      = ", parameter_list_of_values_dictionary_gamma_auto

           # Definging the fit depending on gamma value.
           try:
               logger.info("Running SVM model")
               # Creating a gridsearch object with parameter "C_list_of_values_dictionary"
               internal_cv = GridSearchCV( estimator = SVC(), param_grid = parameter_list_of_values_dictionary_gamma_specified )

           except:
               logger.info("Model failed with gamma = {0} trying automatic gamma "\
                           "instead of.".format(float(args.a)))
               # Creating a gridsearch object with parameter "C_list_of_values_dictionary"
               internal_cv = GridSearchCV( estimator = SVC(), param_grid = parameter_list_of_values_dictionary_gamma_auto )


           # Debugging piece.
           # Performing internal_cv.
           internal_cv.fit(train_data_to_feed, train_classes_to_feed)
           # print "train_classes_to_feed =", train_classes_to_feed
           # print "internal_cv.best_score_ = ",  internal_cv.best_score_
           # print "internal_cv.grid_scores_ = ", internal_cv.grid_scores_
           # print "internal_cv.best_params_['C'] = ", internal_cv.best_params_['C']


           # Performing external_cv using internal_cv
           external_cv = cross_val_score(internal_cv, train_data_to_feed, train_classes_to_feed)
           
	   # Debugging piece.
           # print external_cv
           # print external_cv.mean()
      

           # Checking whether adding this current value to C_list_of_values_current helped improve the result.
	   # For the first run C_list_of_values[0] i.e. for index_current = 0 we assume that external_cv.mean() is the best already.
           # It is the best since we have not tried anything else yet.
           if index_current == 0:
              best_predction_proportion = external_cv.mean()
           
           else:
              # Checking whether adding this extra component helped to what we already had. 
              if external_cv.mean() > best_predction_proportion:
              	 best_predction_proportion = external_cv.mean()
                 C_final = C_list_of_values[index_current]



    
    # This piece of code will work after we decided what C_final we will use.
    # C_final has to be determined at this time via either user of via single or double cv.
    # This number shoul be saved by now in C_final variable.
    # Debugging piece.
    C_final = float(C_final)
    print "The value of C used for the SVM classifier is ", C_final

    
    # Debugging step
    # print "train_classes_to_feed",train_classes_to_feed
    # print "train_data_to_feed",train_data_to_feed


    # Trainig the SVM
    try:
        logger.info("Running SVM model")
        svm_model =  svm.SVC(kernel=args.kernel, C=C_final, gamma=float(args.a), 
                        coef0=float(args.b), degree=int(args.degree))
    except:
        logger.info("Model failed with gamma = {0} trying automatic gamma "\
                    "instead.".format(float(args.a)))
        svm_model =  svm.SVC(kernel=args.kernel, C=C_final, gamma="auto", 
                        coef0=float(args.b), degree=int(args.degree))

    # Fitting the svm_model here.
    svm_model.fit( train_data_to_feed, train_classes_to_feed )


    # Dealing with predicted and classification data frame.

    # Geting predicted values of SVM for the training data set. 
    train_fitted_values = svm_model.predict( train_data_to_feed )

    # Debugging peice.
    # print "train_classes_to_feed = ", train_classes_to_feed
    # print "train_fitted_values   = ", train_fitted_values
    # print "type(train_classes_to_feed) = ", type(train_classes_to_feed)
    # print "type(train_fitted_values)   = ", type(train_fitted_values)
    # print "fitted_values.T.squeeze()   = ", fitted_values.T.squeeze()

    # Converting observed and predicted into pd.series so that we can join them.
    train_fitted_values_series   = pd.Series(train_fitted_values, index=train_classes_to_feed.index )
    train_classes_to_feed_series = pd.Series(train_classes_to_feed,  index=train_classes_to_feed.index )

    # Debugging piece 
    # print "train_fitted_values_series = ", train_fitted_values_series
    # print "train_classes_to_feed_series = ", train_classes_to_feed_series

    # Combining results into the data_frame so that it can be exported.
    classification_df = pd.DataFrame( {'Group_Observed': train_classes_to_feed_series ,
    			               'Group_Predicted': train_fitted_values_series } )
    # Debugging piece
    # print classification_df

    # Outputting classication into the tsv file.
    classification_df.to_csv(args.outClassification, index='sampleID', sep='\t')

    # Computing mismatches between original data and final data
    classification_mismatch_percent = 100 * sum( classification_df['Group_Observed'] == classification_df['Group_Predicted'] )/classification_df.shape[0]
    classification_mismatch_percent_string = str( classification_mismatch_percent ) + ' Percent'
    os.system("echo %s > %s"%( classification_mismatch_percent_string, args.outClassificationAccuracy ) )



    # Geting predicted values of SVM for the target data set. 
    target_fitted_values = svm_model.predict( target_data_to_feed )

    # Debugging peice.
    # print "target_classes_to_feed = ", target_classes_to_feed
    # print "target_fitted_values   = ", target_fitted_values
    # print "type(target_classes_to_feed) = ", type(target_classes_to_feed)
    # print "type(target_fitted_values)   = ", type(target_fitted_values)
    # print "fitted_values.T.squeeze()   = ", fitted_values.T.squeeze()

    # Converting observed and predicted into pd.series so that we can join them.
    target_fitted_values_series   = pd.Series(target_fitted_values, index=target_classes_to_feed.index )
    target_classes_to_feed_series = pd.Series(target_classes_to_feed,  index=target_classes_to_feed.index )

    # Debugging piece 
    # print "target_fitted_values_series = ", target_fitted_values_series
    # print "target_classes_to_feed_series = ", target_classes_to_feed_series

    # Combining results into the data_frame so that it can be exported.
    prediction_df = pd.DataFrame( {'Group_Observed': target_classes_to_feed_series ,
    			           'Group_Predicted': target_fitted_values_series } )

    # Debugging piece
    # print prediction_df

    # Outputting classication into the tsv file.
    prediction_df.to_csv(args.outPrediction, index='sampleID', sep='\t')

    # Computing mismatches between original data and final data
    prediction_mismatch_percent = 100 * sum( prediction_df['Group_Observed'] == prediction_df['Group_Predicted'] )/prediction_df.shape[0]
    prediction_mismatch_percent_string = str( prediction_mismatch_percent ) + ' Percent'
    os.system("echo %s > %s"%( prediction_mismatch_percent_string, args.outPredictionAccuracy ) )


    # Finishing script
    logger.info("Script Complete!")

if __name__ == '__main__':
    # Command line options
    args = getOptions()

    # Turn on logging
    logger = logging.getLogger()
    sl.setLogger(logger)

    # Main
    main(args)
