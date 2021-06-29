import pandas as pd
import numpy as np

def getModelResults(model,feat):
    """
    It gets the results stats out of the model like F,P, mss
    ess, tss, mse, NDF, DDF , R2 , ressids and fitted values.

    :Arguments:
        :type factorNames: stats.model
        :param factorNames: ANOVA model.

    :Returns:
        :rtype preFormula: pd.DataFrame
        :return preFormula: Results table for model results.

    """
    # Getting results
    f      = model.fvalue
    p      = model.f_pvalue
    mss    = model.ssr
    ess    = model.ess
    tss    = mss+ess
    mse    = model.mse_resid
    NDF    = int(model.df_model)
    DDF    = int(model.df_resid)
    R2     = model.rsquared
    resid  = model.resid/np.sqrt(mse)
    fitted = model.fittedvalues
    
    # Puttign al the results together
    anovaResults=[f,p,mss,ess,tss,mse,NDF,DDF,R2]#,resid,fitted]
    
    # Creating indexes for values
    index=["f-Value","p-Value_of_f-Value",
        "ErrorSS","ModelSS","TotalSS",
        "MSE","NDF","DDF", "R2"]#,"resid","fitted"]
    
    # Creating results series
    results = pd.Series(data=anovaResults, index=index)

    # Adding name to this serie of results
    results.name=feat
    
    # Return series
    return results,resid,fitted
