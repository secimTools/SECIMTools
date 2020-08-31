# Import build-in librearies
import copy

# import add-on packages
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


# Importing anova packages
from secimtools.anovaModules.reformatData import reformatData
from secimtools.anovaModules.preProcessing import preProcessing
from secimtools.anovaModules.changeDFOrder import changeDFOrder
from secimtools.anovaModules.flagSignificant import flagSignificant
from secimtools.anovaModules.getModelResults import getModelResults
from secimtools.anovaModules.startANOVAResults import startANOVAResults
from secimtools.anovaModules.generateDinamicCmbs import generateDinamicCmbs
from secimtools.anovaModules.removeAnovaDupResults import removeAnovaDupResults
from secimtools.anovaModules.getModelResultsByGroup import getModelResultsByGroup


def runANOVA(dat, formula, lvlComb, categorical, levels, numerical):
    """
    Core for processing all the ANOVA data.

    :Arguments:
        :type dat: wideToDesign object.
        :param dat: wide, design, group, anno, trans.

        :type formula: dictionary
        :param formula: Contains the formulas in a row:formula fashion.

        :type lvlComb: list.
        :param lvlComb: list with all the levels in the factors.

        :type categorical: list.
        :param categorical: Contains the names of the categorical factors.

        :type levels: list.
        :param levels: Name of the .

        :type numerical: list.
        :param numerical: Contains the names of the numerical factors.

    :Returns:
        :rtype results: pd.DataFrames
        :return results: dataframe in wide format with the results of the model

        :rtype residDat: pd.DataFrames
        :return residDat: Contains the residuals of the model

        :rtype fitDat: pd.DataFrames
        :return fitDat: dataframe with all the fitted data
    """

    # Getting grandMean, variance and mean per groups
    results = startANOVAResults(wide=dat.wide,design=dat.design,groups=categorical)

    # Creating a list of anova results for each metabolite
    full_results     = list()
    resids_list      = list()
    fitted_list      = list()
    significant_list = list()

    # iterating over all the formula keys
    for feat in list(formula.keys()):
        combs=copy.copy(lvlComb)

        # Creating list for fullRes and IndexToDrop
        comb_results = list()
        indexToDrop  = list()

        # Reverse list
        combs.reverse()

        # Take one element of the group and pop it
        while len(combs)>0:
            # Take las element of the list
            elem =  combs.pop()

            # Create tempDF to change Order
            tempDF = changeDFOrder(data=dat.trans, combN=elem, factors=categorical)

            # Running ANOVA on data
	    # AMM changed from .fit_regularized() to .fit()
            anova = ols(formula=formula[feat], data=tempDF).fit()

            # Saving a dataframe for anova results
            group_results = getModelResultsByGroup(anova,levels,numerical)

            # Dropping duplicates
            group_results = removeAnovaDupResults(indexToDrop,df=group_results)

            # Appending results to list
            comb_results.append(group_results)

            # Appending current indexes to indextoDrop list
            indexToDrop= indexToDrop+group_results.index.tolist()

        # Creating one df with all the results
        comb_results = pd.concat(comb_results)

        # Calculating flags for significant pvals
        significant  = flagSignificant(fullRes=comb_results)

        # Getting general results for anova
        model_results,resid,fitted = getModelResults(model=anova,feat=feat)

        # Appending resids and fitted to lists
        resids_list.append(resid)
        fitted_list.append(fitted)

        # Reformating data
        comb_results = reformatData(df=comb_results, feat=feat)
        significant  = reformatData(df=significant, feat=feat)

        # Appending flags to significant flags
        significant_list.append(significant)

        # Concatenating model_results and comb_results and append it
        # to full_results list
        full_results.append(pd.concat([comb_results,model_results]))

    # Concatenating results lists into a dataframe
    full_results = pd.concat(full_results, axis=1)
    significant = pd.concat(significant_list, axis=1)

    # Transpose modelResults dataframe
    full_results = full_results.T
    significant = significant.T

    # Creating pd.Series for Ressids and Fitted Values
    residDat = pd.concat(resids_list, axis=1)
    fitDat   = pd.concat(fitted_list, axis=1)

    # Transpose full_results and concatenate with results
    results = pd.concat([results,full_results], axis=1)

    # Return results
    return results, significant, residDat, fitDat
