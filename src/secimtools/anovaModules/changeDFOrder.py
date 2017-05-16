import pandas as pd
import copy as copy

def changeDFOrder(data,combN,factors):
    """
    Since OLS (ANOVA method) takes the first group as base line we need to 
    change the order of the DF to get all the possible contrasts. This function 
    adds a "1_" before the name of the group, that way we can change the order
    of the groups on the dataframe.

    :Arguments:
        :type data: pd.DataFrame
        :param data: Trans data.

        :type combN: list
        :param combN: unique combinations on the elements from the factors.

        :type factors: list
        :param factors: name of the factors (they should be present on data).

    :Returns:
        :rtype tempDF: pd.DataFrame.
        :return tempDF: Re-Ordered data frame.    

    """
    # Makje a copy of trans
    tempDF = copy.deepcopy(data)

    # Generate new names for groupst this way is possible to change the order 
    #  of  "intercept" in anova
    for elem in combN:
        
        # NewGrpNames for current factor
        newGrpNames = ["0_"+lvl if lvl==elem else lvl for lvl in \
                        tempDF[factors[combN.index(elem)]]]
                        
        # Replace old nams with new names
        tempDF[factors[combN.index(elem)]]=newGrpNames
    
    # Returning results
    return tempDF
