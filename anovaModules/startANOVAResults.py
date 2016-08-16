import pandas as pd
import numpy as np

def startANOVAResults(wide,design,groups):
    """
    This function generates the first elemets on the results table,
    this results have nothing to do with the actual ANOVA method.
    It calculates the mean, variance and the mean by group.
    """
    # Opening results dataframe
    results = pd.DataFrame(index=wide.index)

    # Get grandMean
    results["GrandMean"] = wide.mean(axis=1)

    # Get variance
    results["SampleVariance"]= wide.var(axis=1)

    # Get group means 
    for group in groups:
        for lvlName,subset in design.groupby(group):
            results["Mean {0}-{1}".format(group,lvlName)] = \
            wide[subset.index].mean(axis=1)
            
    # Return results
    return results
