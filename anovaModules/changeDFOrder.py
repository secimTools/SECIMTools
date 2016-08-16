import pandas as pd
import copy as copy

def changeDFOrder(data,combN,factors):
    """
    Since OLS (ANOVA method) takes the first group as base line we need to 
    change the order of the DF to get all the possible contrasts. This function 
    adds a "1_" before the name of the group, that way we can change the order
    of the groups on the dataframe.
    """
    # Makje a copy of trans
    tempDF = copy.deepcopy(data)

    # Generate new names for groupst this way is possible to change the order 
    #  of  "intercept" in anova
    for elem in combN:
        
        # NewGrpNames for current factor
        newGrpNames = ["1_"+lvl if lvl==elem else lvl for lvl in \
                        tempDF[factors[combN.index(elem)]]]
        # Replace old nams with new names
        tempDF[factors[combN.index(elem)]]=newGrpNames
    
    # Returning results
    return tempDF
