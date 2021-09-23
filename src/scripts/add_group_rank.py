#!/usr/bin/env python
################################################################################
# DATE: 2021/March/10
# 
# MODULE: add_group_rank.py
#
# VERSION: 2.0
# 
# AUTHOR: Zihao Liu (zihaoliu@ufl.edu) 
#
# DESCRIPTION: Rank analysis 
#
################################################################################

import pandas as pd
import argparse
import logging
from secimtools.dataManager import logger as sl
from secimtools.dataManager.interface import wideToDesign

def getOptions():
    parser = argparse.ArgumentParser(description="get parameters from Galaxy")
    parser.add_argument('-w', '--wide', dest='wide', required=True, help="The input file in wide format")
    parser.add_argument('-o', '--out', dest='out', required=True, help="Output file name with full path")
    parser.add_argument('-n', '--ngroup', dest='ngroup', default=None, help="Number of bins to rank data in")
    parser.add_argument('-d', '--design', dest='design', required=True, help="A design file assigns which columns to be ranked")
    parser.add_argument('-u', '--uniqID', dest='uniqID', default='rowID', help="Specify the unique row ID column in the wide format input")
    #parser.add_argument('-t', '--tiebreak', dest='tiebreak', required =True, help="Specify the ")
    args=parser.parse_args()
    return args


def main():
    args = getOptions()
    uniId = args.uniqID

    logger = logging.getLogger()
    sl.setLogger(logger)
    logger.info("Importing data with following parameters:"
                "\n\tInput: {0}"
                "\n\tDesign: {1}"
                "\n\tuniqID: {2}".format(args.wide, args.design, args.uniqID))

    logger.info("Importing data through wideToDesign data manager")
    dat = wideToDesign(args.wide, args.design, args.uniqID, 
                        logger=logger)

    df = dat.wide

    if args.ngroup != None:
        groupBins = int(args.ngroup)
        # Changed location of +1 from outside x.rank to outside pd.qcut(), 
        df_rank = df.apply(lambda x: pd.qcut(x.rank(method='first'), q=groupBins, labels = False) + 1, axis=0)
        df_rank.insert(0, uniId, dat.wide.index)
    else:
        #Need to add option for how to break ties
        #Previous code method = 'first', which ranks assigned in order they appear in the array for tie breaks
        df_rank = df.apply(lambda x: x.rank(method='first'), axis = 0)
        df_rank.insert(0, uniId, dat.wide.index)
    df_rank.to_csv(args.out, sep='\t', index = False)


if __name__ == '__main__':
    main()
     
    
    


    
    
    
