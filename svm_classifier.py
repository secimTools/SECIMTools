#!/usr/bin/env python
################################################################################
# DATE: 2017/03/09
#
# SCRIPT: svm_classifier.py
#
# VERSION: 2.0
# 
# AUTHOR: Coded by: Miguel A Ibarra (miguelib@ufl.edu) 
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
from pandas import DataFrame as DF
from pandas import read_csv, read_table
from sklearn import svm

# Import local libraries
from dataManager import logger as sl
from dataManager.interface import wideToDesign

def getOptions(myOpts=None):
    parser = argparse.ArgumentParser( formatter_class=RawDescriptionHelpFormatter)
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
                        help="positive regularization parameter.")
    tool.add_argument('-a',"--a", dest="a", action='store', required=True, 
                        help=" positive coefficient in kernel function.")
    tool.add_argument('-b',"--b", dest="b", action='store', required=True, 
                        help=" independent term coefficient in kernel function.")
    # Tool Output 
    output = parser.add_argument_group(description="Output Paths")
    output.add_argument('-o',"--outfile1", dest="outfile1", action='store', 
                        required=True, help="Output traget set with "\
                        "predicted_class labels.")
    output.add_argument('-acc',"--accuracy_on_training", 
                        dest="accuracy_on_training", action='store',
                         required=True, help="Output accuracy value on the "\
                         "training set.")
    args = parser.parse_args()

    # Standardize paths
    args.outfile1             = os.path.abspath(args.outfile1)
    args.test_wide            = os.path.abspath(args.test_wide)
    args.train_wide           = os.path.abspath(args.train_wide)
    args.test_design          = os.path.abspath(args.test_design)
    args.train_design         = os.path.abspath(args.train_design)
    args.accuracy_on_training = os.path.abspath(args.accuracy_on_training)

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
    # Load test dataset
    test_design=read_table(args.test_design)

    # Loading target dataset trought the interface
    if args.group in test_design.columns:
        target = wideToDesign(wide=args.test_wide,design = args.test_design, 
                            uniqID=args.uniqID, group=args.group, logger=logger)
    else:
        target = wideToDesign(wide=args.test_wide,design = args.test_design, 
                            uniqID=args.uniqID, logger=logger)
     
    # Load training dataset trought the interface
    train = wideToDesign(wide=args.train_wide, design= args.train_design, 
                        uniqID=args.uniqID, group=args.group, logger=logger)
    
    # Dropping missing values
    train.dropMissing()
    train = train.transpose()

    # Dropping missing values
    target.dropMissing()
    target = target.transpose()

    # make sure test and train have the same features
    for i in target.columns:
        if i not in train.columns:
            del target[i]

    #trainig the SVM
    classes=train[args.group].copy()
    del train[args.group]
    try:
        logger.info("Running SVM model")
        model= svm.SVC(kernel=args.kernel, C=float(args.C), gamma=float(args.a), 
                        coef0=float(args.b), degree=int(args.degree))
    except:
        logger.info("Model failed with gamma = {0} trying automatic gamma "\
                    "instead.".format(float(args.a)))
        model= svm.SVC(kernel=args.kernel, C=float(args.C), gamma="auto", 
                        coef0=float(args.b), degree=int(args.degree))
    model.fit(train,classes)

    #predicting classes with the SVM
    if args.group in target.columns:
        del target[args.group]
    try:
        target['predicted_class']=model.predict(target)
    except:
        print "Error: the train set and target set do not appear to have the "\
                "same features (attributes)"
    target.to_csv(args.outfile1,index=False,sep='\t')

    #omputing the accuracy on the training set
    train['predicted_class']=model.predict(train)
    train[args.group]=classes

    accuracy=str(getAccuracy(train)*100)+' percent'
    os.system("echo %s > %s"%(accuracy,args.accuracy_on_training))

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
