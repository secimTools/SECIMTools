#!/usr/bin/env python
###############################################################################
# DATE: 2017/06/29
#
# SCRIPT: svm_classifier.py
#
# VERSION: 2.0
#
# AUTHORS: Ali Ashrafi (a.ali.ashrafi@gmail.com>),
#          Miguel A Ibarra (miguelib@ufl.edu,
#          Alexander Kirpich (akirpich@ufl.edu)
#
# DESCRIPTION: This script flags features based on a given threshold.
#
###############################################################################
import os
import logging
import argparse
from argparse import RawDescriptionHelpFormatter as rdhf
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
from secimtools.dataManager import logger as sl
from secimtools.dataManager.interface import wideToDesign


def getOptions(myOpts=None):
    parser = argparse.ArgumentParser(formatter_class=rdhf)
    standard = parser.add_argument_group(description="Standard Input")
    standard.add_argument('-trw', "--train_wide", dest="train_wide",
                          action='store', required=True,
                          help="wide part of the train dataset.")
    standard.add_argument('-trd', "--train_design", dest="train_design",
                          action='store', required=True, help="design part \
                          of the train dataset.")
    standard.add_argument('-tew', "--test_wide", dest="test_wide",
                          action='store', required=True,
                          help="wide part of the test dataset.")
    standard.add_argument('-ted', "--test_design", dest="test_design",
                          action='store', required=True, help="design part \
                          of the test dataset.")
    standard.add_argument('-g', "--group", dest="group", action='store',
                          required=True, default=False, help="Name of column \
                          in design file with Group/treatment information.")
    standard.add_argument('-id', "--ID", dest="uniqID", action='store',
                          required=True, help="Name of the column with \
                          unique identifiers.")
    tool = parser.add_argument_group(description="Tool Input")
    tool.add_argument('-k', "--kernel", dest="kernel", action='store',
                      required=True, help="choice of kernel function: rbf, \
                      linear, poly, sigmoid.")
    tool.add_argument('-d', "--degree", dest="degree", action='store',
                      required=True, help="(integer) degree for the \
                      polynomial kernel, default 3.")
    tool.add_argument('-c', "--C", dest="C", action='store', required=True,
                      help="positive regularization parameter. This parameter \
                      is ignored when -cv is single or double")
    tool.add_argument('-cv', "--cross_validation", dest="cross_validation",
                      action='store', required=True,
                      help="Choice of cross-validation procedure for the \
                      regularization parameter -c determinantion: none, \
                      single, double.")
    tool.add_argument('-c_lower_bound', "--C_lower_bound",
                      dest="C_lower_bound", action='store', required=False,
                      help="positive regularization parameter lower bound. \
                      Ignored if -cv is none and -c is specified.")
    tool.add_argument('-c_upper_bound', "--C_upper_bound",
                      dest="C_upper_bound", action='store', required=False,
                      help="positive regularization parameter upper bound. \
                      Ignored if -cv is none and -c is specified.")
    tool.add_argument('-a', "--a", dest="a", action='store', required=True,
                      help=" positive coefficient in kernel function.")
    tool.add_argument('-b', "--b", dest="b", action='store', required=True,
                      help=" independent term coefficient in kernel function.")
    output = parser.add_argument_group(description="Output Paths")
    output.add_argument("-oc", "--outClassification", dest="outClassification",
                        action='store', required=True, help="Name of the \
                        output file to store classification performed on the \
                        training data set. TSV format.")
    output.add_argument('-oca', "--outClassificationAccuracy",
                        dest="outClassificationAccuracy", action='store',
                        required=True, help="Output classification accuracy \
                        value on the training data set.")
    output.add_argument("-op", "--outPrediction", dest="outPrediction",
                        action='store', required=True, help="Name of the \
                        output file to store prediction performed on the \
                        target data set. TSV format.")
    output.add_argument('-opa', "--outPredictionAccuracy",
                        dest="outPredictionAccuracy", action='store',
                        required=True, help="Output prediction accuracy value \
                        on the target data set.")
    args = parser.parse_args()
    args.test_wide = os.path.abspath(args.test_wide)
    args.train_wide = os.path.abspath(args.train_wide)
    args.test_design = os.path.abspath(args.test_design)
    args.train_design = os.path.abspath(args.train_design)
    args.outClassification = os.path.abspath(args.outClassification)
    args.outClassificationAccuracy = os.path.abspath(args.outClassificationAccuracy)
    args.outPrediction = os.path.abspath(args.outPrediction)
    args.outPredictionAccuracy = os.path.abspath(args.outPredictionAccuracy)

    return(args)


def correctness(x):
    if x[args.group] == x['predicted_class']:
        return 1
    else:
        return 0


def getAccuracy(data):
    data['correct'] = data.apply(correctness, axis=1)
    accuracy = float(data['correct'].sum())/data.shape[0]
    return accuracy


def main(args):
    target = wideToDesign(wide=args.test_wide, design=args.test_design,
                          uniqID=args.uniqID, group=args.group, logger=logger)
    train = wideToDesign(wide=args.train_wide, design=args.train_design,
                         uniqID=args.uniqID, group=args.group, logger=logger)
    train.wide = train.wide.applymap(float)
    target.wide = train.wide.applymap(float)
    train.dropMissing()
    train = train.transpose()
    target.dropMissing()
    target = target.transpose()
    for i in target.columns:
        if i not in train.columns:
            del target[i]
    cv_status = args.cross_validation
    kernel_final = args.kernel
    gamma_final = float(args.a)
    coef0_final = float(args.b)
    degree_final = int(args.degree)

    # Definging the data to use for the model training.
    train_classes_to_feed = train[args.group].copy()
    train_data_to_feed = train
    del train_data_to_feed[args.group]
    # Definging the data to use for the model target.
    target_classes_to_feed = target[args.group].copy()
    target_data_to_feed = target
    del target_data_to_feed[args.group]
    if cv_status == "none":
        logger.info(u"Using the value of C specified by the user.")
        C_final = float(args.C)
    if cv_status == "single":
        logger.info(u"Using the value of C determined via a single \
                    cross-validation.")
        if (len(train_classes_to_feed) < 100):
            logger.info(u"The required number of samples for a single \
                        cross-validation procedure is at least 100. The \
                        dataset has {0}.".format(len(train_classes_to_feed)))
            logger.info(u"Exiting the tool.")
            exit(1)
        C_lower = float(args.C_lower_bound)
        C_upper = float(args.C_upper_bound)
        C_list_of_values = np.linspace(C_lower, C_upper, 20)
        gamma_param_dict = {"kernel": [kernel_final], "C": C_list_of_values,
                            "gamma":  [gamma_final], "coef0": [coef0_final],
                            "degree": [degree_final]}
        auto_gamma_param_dict = {"kernel": [kernel_final],
                                 "C": C_list_of_values, "gamma":  ["auto"],
                                 "coef0": [coef0_final],
                                 "degree": [degree_final]}
        try:
            logger.info("Running SVM model")
            internal_cv = GridSearchCV(estimator=SVC(),
                                       param_grid=gamma_param_dict)
        except ValueError:
            logger.info("Model failed with gamma = {0} trying automatic gamma \
                        instead of.".format(float(args.a)))
            internal_cv = GridSearchCV(estimator=SVC(),
                                       param_grid=auto_gamma_param_dict)
        internal_cv.fit(train_data_to_feed, train_classes_to_feed)
        C_final = internal_cv.best_params_['C']
    if cv_status == "double":
        logger.info(u"Using the value of C determined via a double \
                    cross-validation.")
        if (len(train_classes_to_feed) < 100):
            logger.info(u"The required number of samples for a double \
                        cross-validation procedure is at least 100. The \
                        dataset has {0}.".format(len(train_classes_to_feed)))
            logger.info(u"Exiting the tool.")
            exit()
        C_lower = float(args.C_lower_bound)
        C_upper = float(args.C_upper_bound)
        C_list_of_values = np.linspace(C_lower, C_upper, 20)
        C_final = C_list_of_values[0]
        for index_current in range(0, 20):
            C_list_of_values_current = np.linspace(C_list_of_values[0],
                    C_list_of_values[index_current], (index_current+1))
            # Creating dictionary for the single cross-validation procedure.
            # In this dictionary gamma is speficied by the user.
            gamma_param_dict = {"kernel": [kernel_final],
                                "C": C_list_of_values_current,
                                "gamma": [gamma_final],
                                "coef0":  [coef0_final],
                                "degree": [degree_final]}
            # gamma is determined automatically if the first dictionary fails.
            auto_gamma_param_dict = {"kernel": [kernel_final],
                                     "C": C_list_of_values_current,
                                     "gamma":  ["auto"],
                                     "coef0":  [coef0_final],
                                     "degree": [degree_final]}
            try:
                logger.info("Running SVM model")
                internal_cv = GridSearchCV(estimator=SVC(),
                                           param_grid=gamma_param_dict)
            except ValueError:
                logger.info("Model failed with gamma = {0} trying automatic \
                            gamma instead of.".format(float(args.a)))
                internal_cv = GridSearchCV(estimator=SVC(),
                                           param_grid=auto_gamma_param_dict)
            internal_cv.fit(train_data_to_feed, train_classes_to_feed)
            external_cv = cross_val_score(internal_cv, train_data_to_feed,
                                          train_classes_to_feed)
            if index_current == 0:
                best_predction_proportion = external_cv.mean()
            else:
                if external_cv.mean() > best_predction_proportion:
                    best_predction_proportion = external_cv.mean()
                    C_final = C_list_of_values[index_current]
    C_final = float(C_final)
    print("The value of C used for the SVM classifier is {}".format(C_final))
    try:
        logger.info("Running SVM model")
        svm_model = svm.SVC(kernel=args.kernel, C=C_final, gamma=float(args.a),
                            coef0=float(args.b), degree=int(args.degree))
    except ValueError:
        logger.info("Model failed with gamma = {0} trying automatic gamma \
                    instead.".format(float(args.a)))
        svm_model = svm.SVC(kernel=args.kernel, C=C_final, gamma="auto",
                            coef0=float(args.b), degree=int(args.degree))
    svm_model.fit(train_data_to_feed, train_classes_to_feed)
    train_fitted_values = svm_model.predict(train_data_to_feed)
    train_fitted_values_series = pd.Series(train_fitted_values,
                                           index=train_classes_to_feed.index)
    train_classes_to_feed_series = pd.Series(train_classes_to_feed,
                                             index=train_classes_to_feed.index)
    classification_df = pd.DataFrame({'Group_Observed': train_classes_to_feed_series,
                                      'Group_Predicted': train_fitted_values_series})
    classification_df.to_csv(args.outClassification, index='sampleID', sep='\t')
    classification_mismatch_percent = 100 * sum(classification_df['Group_Observed'] == classification_df['Group_Predicted'])/classification_df.shape[0]
    classification_mismatch_percent_string = str(classification_mismatch_percent) + ' Percent'
    os.system("echo {0} > {1}".format(classification_mismatch_percent_string, args.outClassificationAccuracy))
    target_fitted_values = svm_model.predict(target_data_to_feed)
    target_fitted_values_series = pd.Series(target_fitted_values, index=target_classes_to_feed.index)
    target_classes_to_feed_series = pd.Series(target_classes_to_feed, index=target_classes_to_feed.index)
    prediction_df = pd.DataFrame({'Group_Observed': target_classes_to_feed_series,
                                  'Group_Predicted': target_fitted_values_series})
    prediction_df.to_csv(args.outPrediction, index='sampleID', sep='\t')
    prediction_mismatch_percent = 100 * sum(prediction_df['Group_Observed'] == prediction_df['Group_Predicted'])/prediction_df.shape[0]
    prediction_mismatch_percent_string = str(prediction_mismatch_percent) + ' Percent'
    os.system("echo {0} > {1}".format(prediction_mismatch_percent_string,
                                      args.outPredictionAccuracy))
    logger.info("Script Complete!")


if __name__ == '__main__':
    args = getOptions()
    logger = logging.getLogger()
    sl.setLogger(logger)
    main(args)

