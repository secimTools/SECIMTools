# Import built-in modules
import re

# Import Add-on modules
import pandas as pd
import numpy as np

# Import ANOVA Modules
from secimtools.anovaModules.gimmeTheMissin import gimmeTheMissin

def getModelResultsByGroup(model,levels, numerical):
    """
    This functiuon generates the results by group for and ANOVA
    object (model). The results that it generates includes:
     -  Coeficients
     -  Std. error
     -  T values
     -  Probability for the t-values

    :Arguments:
        :type model: statsmodel.ols
        :param model: ANOVA model

        :type levels: list
        :param levels: groups inside a factor.

        :type numerical: list
        :param numerical: Numerical factor(s) if any.

    :Returns:
        :rtype preFormula: list
        :return preFormula: List withh all the missing values.
    """
    # Extracting the parameters we are interested in from ANOVA
    # These values are going to be used multiple times
    coef = -(model.params)
    stde = model.bse
    t    = -(model.tvalues)
    pt   = model.pvalues
    log  = -np.log10(model.pvalues)
        
    #Add name to previous series
    t.name    ="t-Value_for_Diff"
    stde.name ="stdError_for_Diff"
    coef.name ="diff_of"
    pt.name   ="prob_greater_than_t_for_diff"
    log.name  ="-log10_p-value_"


    # Concat all dataframes
    df = pd.concat([coef,stde,t,pt,log],axis=1)
    
    # Removing intercepts
    df.drop("Intercept",inplace=True,axis="index")
    
    # Removing numerical factors
    for numeric in numerical:
        if numeric in df.index.tolist():
            df.drop(numeric,inplace=True,axis="index") 
    
    # New Index Names
    newIndexNames = {origIndx:re.sub(".+\[T\.|\]","",origIndx)for origIndx in df.index.tolist()}
    
    # Rename df indexes with new Indexes names
    df.rename(newIndexNames,inplace=True)
        
    #Getting the baseline 
    baseLines = gimmeTheMissin(df.index.tolist(),levels)
        
    # Creating pretty names for indexes
    oldIndex = dict()
    for origIndx,base in zip(df.index.tolist(),baseLines):
        if base == origIndx:
            df.drop(origIndx,inplace=True)
        else:
            oldIndex[origIndx] = "{0}-{1}".format(base,origIndx)

    # Creating 
    df.replace(-0,np.nan, inplace=True)
    df.replace(0,np.nan, inplace=True)
    
    #Rename indexs
    df.rename(index=oldIndex, inplace=True)

    #Returns
    return df
