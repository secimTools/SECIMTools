#!/usr/bin/env python
######################################################################################
# DATE: 2016/August/10
# 
# MODULE: anova_lm.py
#
# VERSION: 1.0
# 
# AUTHOR: Miguel Ibarra (miguelib@ufl.edu) ed. Matt Thoburn (mthoburn@ufl.edu) 
#
# DESCRIPTION: This tool runs a multiway ANOVA on wide data
#
#######################################################################################
# Built-in packages
import re
import copy
import logging
import argparse
import itertools
from collections import defaultdict
from argparse import RawDescriptionHelpFormatter

# Add-on packages
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

# Local Packages
import logger as sl
from interface import wideToDesign

# Importing anova packages
from anovaModules.qqPlot import qqPlot
from anovaModules.volcano import volcano
from anovaModules.runANOVA import runANOVA
from anovaModules.preProcessing import preProcessing
from anovaModules.generateDinamicCmbs import generateDinamicCmbs

def getOptions():
    """ Function to pull in arguments """
    description = """ One-Way ANOVA """
    parser = argparse.ArgumentParser(description=description,
                             formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('-i',"--input", dest="input", action='store', 
            required=True, help="Input dataset in wide format.")
    parser.add_argument('-d',"--design", dest="design", action='store', 
            required=True, help="Design file.")
    parser.add_argument('-id',"--ID", dest="uniqID", action='store', 
            required=True, help="Name of the column with unique identifiers.")
    parser.add_argument('-f',"--factors", dest="factors", action='store', 
            required=True, help="Factors to run ANOVA")
    parser.add_argument('-t',"--factorTypes", dest="ftypes", action='store', 
            required=True, help="Type of factors to run ANOVA")
    parser.add_argument('-in',"--interactions", dest="interactions", action="store_true", 
            required=False, help="Ask for interactions to run ANOVA")
    parser.add_argument('-o',"--out", dest="oname", action="store", 
            required=True, help="Output file name.")
    parser.add_argument('-f1',"--fig", dest="ofig", action="store", 
            required=True, help="Output figure name for q-q plots [pdf].")
    parser.add_argument('-f2',"--fig2", dest="ofig2", action="store", 
            required=True, help="Output figure name for volcano plots [pdf].")
    args = parser.parse_args()

    return(args)

def main(args):
    # Import data
    dat = wideToDesign(args.input,args.design,args.uniqID)

    # Generate formula Formula
    preFormula,categorical,numerical,levels = preProcessing(design=dat.design,
                        factorTypes=args.ftypes, factorNames=args.factors)

    # Transpose data
    dat.trans  = dat.transpose()

    # Create combination of groups
    nLevels =  [list(itertools.chain.from_iterable(levels))]
    reverseLevels = copy.copy(nLevels)
    reverseLevels.reverse()
    lvlComb = list()
    generateDinamicCmbs(reverseLevels,lvlComb)

    # Maps every metabolite to its formulas
    dictFormula = {feature:"{0}~{1}".format(str(feature),preFormula) for feature \
                    in dat.wide.index.values}

    # running anova
    results,residDat,fitDat = runANOVA(dat=dat, categorical=categorical,
                            levels=levels, lvlComb=lvlComb, formula=dictFormula, 
                            numerical=numerical)

    #OUTPUT!!
    # QQ plots    
    logger.info('Generating q-q plots.')
    qqPlot(residDat.T, fitDat.T, args.ofig)

    # Generate Volcano plots
    logger.info('Generating volcano plots.')
    volcano(lvlComb, results, args.ofig2)

    # Round results to 4 digits and save
    results = results.round(4)
    results.to_csv(args.oname, sep="\t")

if __name__ == '__main__':
    # Command line options
    args = getOptions()

    logger = logging.getLogger()
    sl.setLogger(logger)

    logger.info(u"""Importing data with following parameters: \
        \n\tWide: {0}\
        \n\tDesign: {1}\
        \n\tUnique ID: {2}\
        \n\tFactors: {3}"""
        .format(args.input, args.design, args.uniqID, args.factors))

    main(args)