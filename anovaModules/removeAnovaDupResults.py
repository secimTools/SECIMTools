import pandas as pd

def removeAnovaDupResults(toDrop,df):
    """
    Removes the duplicate results on the final table by deleting the duplicates
    ie. a-b == than b-a then we delete b-a.

    :Arguments:
        :type toDrop: list
        :param toDrop: Pairs of metabolites previously reported.

        :type df: pandas.DataFrame.
        :param df: Original data.

    :Returns:
        :rtype df: pandas.DataFrame.
        :return df: Data with out duplicates
    """
    # For every index in the df list
    for idx in df.index.tolist():
        
        # Split index using "-"
        idxElements =  idx.split("-")
        
        # For element to drop in toDrop
        for drop in toDrop:
            
            # Split element to drop by "-"
            d1,d2 = drop.split("-")
            
            # If element to drop is in df index
            if (d1 in idxElements) and (d2 in idxElements):
                
                # Drop df row on that index
                df = df.drop(idx,axis="index")
                
    # Return df
    return df