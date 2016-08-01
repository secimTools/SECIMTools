#!/usr/bin/env python
# Built-in packages
import logging
import argparse
from argparse import RawDescriptionHelpFormatter

# Add-on packages
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Local Packages
from interface import wideToDesign
import logger as sl


def getOptions(myopts=None):
    """ Function to pull in arguments """
    description = """ One-Way ANOVA """
    parser = argparse.ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--input", dest="fname", action='store', required=True, help="Input dataset in wide format.")
    parser.add_argument("--design", dest="dname", action='store', required=True, help="Design file.")
    parser.add_argument("--ID", dest="uniqID", action='store', required=True, help="Name of the column with unique identifiers.")
    parser.add_argument("--group", dest="group", action='store', required=True, help="Group/treatment identifier in design file.")
    parser.add_argument("--num", dest="num", action='store', type=int, required=True, default=1000, help="Number of estimators.")
    parser.add_argument("--out", dest="oname", action='store', required=True, help="Output file name.")
    parser.add_argument("--out2", dest="oname2", action='store', required=True, help="Output file name.")
    parser.add_argument("--debug", dest="debug", action='store_true', required=False, help="Add debugging log output.")

    if myopts:
        args = parser.parse_args(myopts)
    else:
        args = parser.parse_args()

    return(args)


def main(args):
    # Import data and transpose
    logger.info(u'Importing data with following parameters: \n\tWide: {0}\n\tDesign: {1}\n\tUnique ID: {2}\n\tGroup Column: {3}'.format(args.fname, args.dname, args.uniqID, args.group))
    dat = wideToDesign(args.fname, args.dname, args.uniqID, args.group, clean_string=True)
    data = dat.transpose()
    data.dropna(axis=1, inplace=True)

    # Pull classifications out of dataset
    classes = data[dat.group].copy()
    data.drop(dat.group, axis=1, inplace=True)
    #TODO: Random forest does not handle NaNs, need to figure out the proper way to impute values.

    # Build Random Forest classifier
    logger.info('Creating classifier')
    model = RandomForestClassifier(n_estimators=args.num)
    model.fit(data, classes)

    # Identify features
    importance = pd.DataFrame([data.columns, model.feature_importances_]).T.sort(columns=1, ascending=False)

    # Export features ranked by importance
    logger.info('Exporting features')
    rev = importance.applymap(lambda x: dat.revertStr(x))
    rev.columns = ('feature', 'ranked_importance')
    rev.to_csv(args.oname2, index=False, sep='\t')

    # Select data based on features
    data = data[importance.ix[:, 0].tolist()]
    selected_data = pd.DataFrame(model.transform(data, threshold=0))
    selected_data.columns = [dat.revertStr(x) for x in data.columns]

    # Merge on classes and export
    logger.info('Exporting transformed data')
    clDf = pd.DataFrame(classes)
    clDf.reset_index(inplace=True)
    out = clDf.join(selected_data)
    out.to_csv(args.oname, index=False, sep='\t', float_format="%.4f")


if __name__ == '__main__':
    # Command line options
    args = getOptions()

    # Activate Logger
    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
    else:
        sl.setLogger(logger)

    main(args)

