# Built-in packages
import os, sys
import argparse
from argparse import RawDescriptionHelpFormatter

# Add-on packages
import numpy as np
from pandas import DataFrame as DF
from pandas import read_csv, read_table
from sklearn import svm
from interface import wideToDesign

def getOptions(myOpts=None):
    parser = argparse.ArgumentParser( formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('-trw',"--train_wide", dest="train_wide", action='store', required=True, help="wide part of the train dataset.")
    parser.add_argument('-trd',"--train_design", dest="train_design", action='store', required=True, help="design part of the train dataset.")
    parser.add_argument('-tew',"--test_wide", dest="test_wide", action='store', required=True, help="wide part of the test dataset.")
    parser.add_argument('-ted',"--test_design", dest="test_design", action='store', required=True, help="design part of the test dataset.")
    parser.add_argument('-g',"--class_column_name", dest="class_column_name", action='store', required=True, help="Name of column in design file with Group/treatment information.")
    parser.add_argument('-id',"--ID", dest="uniqID", action='store', required=True, help="Name of the column with unique identifiers.")
    parser.add_argument('-k',"--kernel", dest="kernel", action='store', required=True, help="choice of kernel function: rbf, linear, poly, sigmoid.")
    parser.add_argument('-d',"--degree", dest="degree", action='store', required=True, help="(integer) degree for the polynomial kernel, default 3.")
    parser.add_argument('-c',"--C", dest="C", action='store', required=True, help="positive regularization parameter.")
    parser.add_argument('-a',"--a", dest="a", action='store', required=True, help=" positive coefficient in kernel function.")
    parser.add_argument('-b',"--b", dest="b", action='store', required=True, help=" independent term coefficient in kernel function.")
    parser.add_argument('-o',"--outfile1", dest="outfile1", action='store', required=True, help="Output traget set with predicted_class labels.")
    parser.add_argument('-acc',"--accuracy_on_training", dest="accuracy_on_training", action='store', required=True, help="Output accuracy value on the training set.")

    if myopts:
        args = parser.parse_args(myopts)
    else:
        args = parser.parse_args()

    return(args)

def correctness(x):
    if x[class_column_name]==x['predicted_class']:
        return 1
    else:
        return 0

def accuracy(data):
    data['correct']=data.apply(correctness,axis=1)
    accuracy=float(data['correct'].sum())/data.shape[0]
    return accuracy

def main(args):

    train = wideToDesign(wide=args.train_wide, design= args.train_design, uniqID=args.uniqID, group=args.class_column_name).transpose()

    test_design=read_table(args.test_design)
    if args.class_column_name in test_design.columns:
        target = wideToDesign(wide=args.test_wide,design = args.test_design, uniqID=args.uniqID, group=args.class_column_name).transpose()
    else:
        target = wideToDesign(wide=args.test_wide,design=args.test_design, uniqID=args.uniqID).transpose()

    # make sure test and train have the same features
    for i in target.columns:
        if i not in train.columns:
            del target[i]

    #trainig the SVM
    class_column_name = args.class_column_name
    classes=train[class_column_name].copy()
    del train[class_column_name]
    model= svm.SVC(kernel=args.kernel, C=float(args.C), gamma=float(args.a), coef0=float(args.b), degree=int(args.degree))
    model.fit(train,classes)

    #predicting classes with the SVM
    if class_column_name in target.columns:
        del target[class_column_name]
    try:
        target['predicted_class']=model.predict(target)
    except:
        print "Error: the train set and target set do not appear to have the same features (attributes)"
    target.to_csv(args.outfile1,index=False,sep='\t')

    #omputing the accuracy on the training set
    train['predicted_class']=model.predict(train)
    train[class_column_name]=classes


    accuracy=str(accuracy(train)*100)+' percent'
    os.system("echo %s > %s"%(accuracy,args.accuracy_on_training))

if __name__ == '__main__':
    # Command line options
    args = getOptions()

    # Main
    main(args)
