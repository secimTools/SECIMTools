#!/usr/bin/env python
##
##  Created on Wed May 12 11:10:16 2021
##
##  @author: zach
##

import sys, os
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import logging
from secimtools.dataManager import logger as sl
from secimtools.dataManager.interface import wideToDesign
import argparse

r = ro.r
ro.r.source(os.path.dirname(sys.argv[0]) + '/metafor_wrappers.R')


def getoptions():
    parser = argparse.ArgumentParser(description="Perform meta-analysis by metafor in R")
    parser.add_argument("-w", "--wide", dest="wide", required=True, help="The input wide file")
    parser.add_argument("-d", "--design", dest="design", required=True, help="The input design file")
    parser.add_argument('-id', '--uniqID', dest='uniqID', default='rowID', help="Specify the unique row ID column in the wide format input")
    parser.add_argument("-s", "--study", dest="study", required=True, help="The column name in the design file to specify study")
    parser.add_argument("-t", "--treatment", dest="treatment", required=True, help="The column name in the design file to specify treatment")
    parser.add_argument("-c", "--contrast", dest="contrast", required=True, help="The contrast to be used for the meta-anlysis")
    parser.add_argument("-fr", "--forest", dest="forest", default=None, help="The forest plot output directory plus prefix")
    parser.add_argument("-o", "--summary", dest="summary", required=True, help="The analysis summary file output name and path")
    parser.add_argument("-r", "--report", dest="report", required=True, help="The analysis report file output name and path")
    parser.add_argument("-m", "--model", dest="model", default="FE", help="The meta-analysis model that will be applied")
    parser.add_argument("-es", "--effectSize", dest="effectSize", default="SMD", help="The approach used to calculate the effect size, default is SMD")
    parser.add_argument("-cm", "--cmMethod", dest="cmMethod", default = 'UB', help="The method used to compute the sampling variances, default is unbiased estimation, can be 'LS' and 'AV' etc.")
    parser.add_argument("-bg", "--background", dest="background", default = False, help="whether each factor will compare to all the controls in a set")
    args = parser.parse_args()
    return(args)

def main():
    saveStdout = sys.stdout
    args = getoptions()
    logger = logging.getLogger()
    sl.setLogger(logger)
    logger.info("Importing data with following parameters:"
                "\n\tInput: {0}"
                "\n\tDesign: {1}"
                "\n\tuniqID: {2}".format(args.wide, args.design, args.uniqID))

    logger.info("Importing data through wideToDesign data manager")  
    
    effect = []
    se = []
    z_value = []
    p_value = []
    ci_low = []
    ci_upper = []
    summary = [effect, se, z_value, p_value, ci_low, ci_upper]
    

    dat = wideToDesign(args.wide, args.design, args.uniqID, 
                        logger=logger)


    dat.trans  = dat.transpose()
    
    ## subset here based on user provided mutant and set (to pull relevant pd1074)

    contrast = args.contrast.split(",")    
    with localconverter(ro.default_converter + pandas2ri.converter):
        data_rform = ro.conversion.py2rpy(dat.trans)
        rcontrast = ro.conversion.py2rpy(contrast)

    features  = dat.wide.index.values
    with open(args.report, "w+") as f:
        sys.stdout = f
        for fea in features:
            print("\n\n\n===============================================\ntest feature: " + fea)
            if args.forest:
                outfig = args.forest + "_" + fea + "_" + args.model + "_forest.pdf"
                #outfig = args.forest + "/" + fea + "_" + args.model + "_forest.pdf"
            else:
                outfig = 'NOFIG'
            res_fromR = r.meta_batchCorrect(data = data_rform, 
                                      dependent = fea, 
                                      study = args.study, 
                                      treatment = args.treatment,
                                      factors = rcontrast,
                                      forest = outfig, 
                                      myMethod = args.model, 
                                      myMeasure = args.effectSize,
                                      myvtype = args.cmMethod,
                                      toBackground = args.background)
        
            with localconverter(ro.default_converter + pandas2ri.converter):
                res = ro.conversion.rpy2py(res_fromR)

            for i, j in zip(res, summary):
                j.append(i)
    sys.stdout = saveStdout
    df = pd.DataFrame({"effect" : effect,
                       "se" : se,
                       "z_value" : z_value,
                       "p_value" : p_value,
                       "ci_low" : ci_low,
                       "ci_upper" : ci_upper})
    
    df.index = features
    df.index.name = "featureID"
    df.to_csv(args.summary)
    return

if __name__ == "__main__":
    main()

