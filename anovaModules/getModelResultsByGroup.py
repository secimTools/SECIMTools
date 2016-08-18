# Import built-in modules
import re

# Import Add-on modules
import pandas as pd
import numpy as np

# Import ANOVA Modules
from anovaModules.gimmeTheMissin import gimmeTheMissin

def getModelResultsByGroup(model,levels, numerical):
    """
    This functiuon generates the results by group for and ANOVA
    object (model). The results that it generates includes:
     -  Coeficients
     -  Std. error
     -  T values
     -  Probability for the t-values 

    """
    # Extracting the parameters we are interested in from ANOVA
    # These values are going to be used multiple times
    coef = -(model.params)
    stde = model.bse
    t    = model.tvalues
    pt   = model.pvalues
        
    #Add name to previous series
    t.name    ="t-Value for Diff"
    stde.name ="StdError for Diff"
    coef.name ="Diff of"
    pt.name   ="Prob>|t| for Diff"

    # Concat all dataframes
    df = pd.concat([coef,stde,t,pt],axis=1)
    
    # Removing intercepts
    df.drop("Intercept",inplace=True,axis="index")

    # Removing numerical factors
    for numeric in numerical:
        if numeric in df.index.values:
            df.drop(numeric,inplace=True,axis="index") 
    
    # New Index Names
    newIndexNames = {origIndx:re.sub(".+\[T\.|\]","",origIndx)for origIndx in df.index.values}
    
    # Rename df indexes with new Indexes names
    df.rename(newIndexNames,inplace=True)
        
    #Getting the baseline 
    baseLines = gimmeTheMissin(df.index.values,levels)
        
    # Creating pretty names for indexes
    oldIndex = dict()
    for base,origIndx in zip(baseLines, df.index.values):
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