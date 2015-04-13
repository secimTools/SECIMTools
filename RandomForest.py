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


def getOptions():
    """ Function to pull in arguments """
    description = """ One-Way ANOVA """
    parser = argparse.ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--input", dest="fname", action='store', required=True, help="Input dataset in wide format.")
    parser.add_argument("--design", dest="dname", action='store', required=True, help="Design file.")
    parser.add_argument("--ID", dest="uniqID", action='store', required=True, help="Name of the column with unique identifiers.")
    parser.add_argument("--group", dest="group", action='store', required=True, help="Group/treatment identifier in design file.")
    parser.add_argument("--num", dest="num", action='store', type=int, required=True, help="Number of estimators.")
    parser.add_argument("--out", dest="oname", action='store', required=True, help="Output file name.")
    parser.add_argument("--out2", dest="oname2", action='store', required=True, help="Output file name.")
    parser.add_argument("--debug", dest="debug", action='store_true', required=False, help="Add debugging log output.")
#     args = parser.parse_args()
    args = parser.parse_args(['--input', '/home/jfear/sandbox/secim/data/ST000015.tsv',
                              '--design', '/home/jfear/sandbox/secim/data/ST000015_design.tsv',
                              '--ID', 'Name',
                              '--group', 'treatment',
                              '--num', '10',
                              '--out', '/home/jfear/sandbox/secim/data/test.csv',
                              '--out2', '/home/jfear/sandbox/secim/data/test2.csv',
                              '--debug'])
    return(args)


def main(args):
    # Import data
    logger.info(u'Importing data with following parameters: \n\tWide: {0}\n\tDesign: {1}\n\tUnique ID: {2}\n\tGroup Column: {3}'.format(args.fname, args.dname, args.uniqID, args.group))
    dat = wideToDesign(args.fname, args.dname, args.uniqID, args.group, clean_string=True)
    trans = dat.transpose()

    model = RandomForestClassifier(n_estimators=args.num)
    import ipdb; ipdb.set_trace()
    model.fit(trans[dat.wide.index], trans[dat.group])

    importance = pd.DataFrame([trans.columns, model.feature_importances_]).T.sort(columns=1, ascending=False)
    importance.to_csv(args.oname2, index=False, header=False, sep='\t')

    data = data[importance.ix[:,0].tolist()]
    selected_data = DF(model.transform(data, threshold=0))
    selected_data.columns = data.columns
    selected_data[class_column_name] = classes
    selected_data.to_csv(args.oname, index=False, sep='\t')

if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
    else:
        sl.setLogger(logger)

    main(args)
