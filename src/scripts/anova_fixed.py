#!/usr/bin/env python
######################################################################################
# DATE: 2016/August/10
# 
# MODULE: anova_fixed.py
#
# VERSION: 1.0
# 
# AUTHOR: Miguel Ibarra (miguelib@ufl.edu) ed. Matt Thoburn (mthoburn@ufl.edu) 
#
# DESCRIPTION: This tool runs a fixed effects multiway ANOVA on wide data.
#
#######################################################################################
# Import built-in packages
import re
import os
import copy
import logging
import argparse
import itertools
from collections import defaultdict
from argparse import RawDescriptionHelpFormatter

# Import add-on packages
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

# Import local data libraries
from secimtools.dataManager import logger as sl
from secimtools.dataManager.interface import wideToDesign

# Import local plotting libraries
from secimtools.anovaModules.qqPlot import qqPlot
from secimtools.anovaModules.volcano import volcano
from secimtools.anovaModules.runANOVA import runANOVA
from secimtools.anovaModules.preProcessing import preProcessing
from secimtools.anovaModules.generateDinamicCmbs import generateDinamicCmbs
 
def getOptions():
    """ Function to pull in arguments """
    description = """ ANOVA Fixed Effects"""
    parser = argparse.ArgumentParser(description=description,
                             formatter_class=RawDescriptionHelpFormatter)
    # Standard input
    standard = parser.add_argument_group(description="Required Input")
    standard.add_argument('-i',"--input", dest="input", action='store', 
            required=True, help="Input dataset in wide format.")
    standard.add_argument('-d',"--design", dest="design", action='store', 
            required=True, help="Design file.")
    standard.add_argument('-id',"--ID", dest="uniqID", action='store', 
            required=True, help="Name of the column with unique identifiers.")
    standard.add_argument('-f',"--factors", dest="factors", action='store', 
            required=True, help="Factors to run ANOVA")
    # Tool input
    tool = parser.add_argument_group(description="Tool Input")
    tool.add_argument('-t',"--factorTypes", dest="ftypes", action='store', 
            required=True, help="Type of factors to run ANOVA")
    tool.add_argument('-in',"--interactions", dest="interactions", action="store_true", 
            required=False, help="Ask for interactions to run ANOVA")
    # Tool output
    output = parser.add_argument_group(description="Output")
    output.add_argument('-o',"--out", dest="oname", action="store", 
            required=True, help="Output file name.")
    output.add_argument('-fl',"--flags", dest="flags", action="store", 
            required=True, help="Flags file name.")
    output.add_argument('-f1',"--fig", dest="ofig", action="store", 
            required=True, help="Output figure name for q-q plots [pdf].")
    output.add_argument('-f2',"--fig2", dest="ofig2", action="store", 
            required=True, help="Output figure name for volcano plots [pdf].")
    args = parser.parse_args()

    # Standardized paths
    args.ofig  = os.path.abspath(args.ofig)
    args.ofig2 = os.path.abspath(args.ofig2)
    args.oname = os.path.abspath(args.oname)
    args.flags = os.path.abspath(args.flags)

    return(args)

def main(args):
    # Import data
    dat = wideToDesign(args.input,args.design,args.uniqID,logger=logger)

    # Cleaning from missing data
    dat.dropMissing()

    # Generate formula Formula
    preFormula,categorical,numerical,levels,dat.design = preProcessing(design=dat.design,
                        factorTypes=args.ftypes, factorNames=args.factors)

    # Transpose data
    dat.trans  = dat.transpose()

    # if interactions
    if args.interactions:
        logger.info("Running ANOVA on interactions")
        dat.trans["_treatment_"] = dat.trans.apply(lambda x: \
                                "_".join(map(str,x[categorical].values)),axis=1)

        dat.design["_treatment_"] = dat.design.apply(lambda x: \
                                "_".join(map(str,x[categorical].values)),axis=1)

        # if numerical adde then to the formula
        if len(numerical)>0:
            formula = ["C(_treatment_)"]+numerical
        else:
            formula = ["C(_treatment_)"]

        # Concatenatig the formula
        formula = "+".join(formula)

        # Getting new formula for interactions
        dictFormula = {feature:"{0}~{1}".format(str(feature),formula) \
                    for feature in dat.wide.index.tolist()}

        # Creating levelCombinations
        levels=sorted(list(set(dat.trans["_treatment_"].tolist())))

        # Creating levelCombinations
        reverseLevels = copy.copy(levels)
        reverseLevels.reverse()
        lvlComb = list()
        generateDinamicCmbs([levels],lvlComb)

        # Running anova
        logger.info('Running anova models')
        results,significant,residDat,fitDat = runANOVA(dat=dat, categorical=["_treatment_"],
                                levels=[levels], lvlComb=lvlComb, formula=dictFormula, 
                                numerical=numerical)
    else:
        logger.info("Running ANOVA without interactions")
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
        logger.info('Running anova models')
        results,significant,residDat,fitDat = runANOVA(dat=dat, categorical=categorical,
                                levels=levels, lvlComb=lvlComb, formula=dictFormula, 
                                numerical=numerical)

    # QQ plots    
    logger.info('Generating q-q plots.')
    qqPlot(residDat.T, fitDat.T, args.ofig)

    # Generate Volcano plots
    logger.info('Generating volcano plots.')
    volcano(lvlComb, results, args.ofig2)

    # Round results to 4 digits and save
    results = results.round(4)
    results.index.name = dat.uniqID
    results.to_csv(args.oname, sep="\t")

    # Flags
    significant.index.name = dat.uniqID
    significant.to_csv(args.flags, sep="\t")

if __name__ == '__main__':
    # Command line options
    args = getOptions()

    # Setting up logger
    logger = logging.getLogger()
    sl.setLogger(logger)
    logger.info("""Importing data with following parameters: \
        \n\tWide: {0}\
        \n\tDesign: {1}\
        \n\tUnique ID: {2}\
        \n\tFactors: {3}"""
        .format(args.input, args.design, args.uniqID, args.factors))

    # Calling main script
    main(args)
