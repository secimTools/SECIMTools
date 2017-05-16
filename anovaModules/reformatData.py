# Importing packages
import pandas as pd

def reformatData(df, feat, hasCombinations=True):
    """
    Reformats data from stacked data to pandas dataframes.
    """
    # Initializing variables
    new_data  = list()
    new_index = list()

    # Stacking data
    stacked = df.T.stack(level=0)

    # Iteratting over index on stacked data one for the type of data and other 
    # for the combination. This will slice the data and generate new indexes.
    for name,combination in stacked.index:
        new_index.append("{0}_{1}".format(name,combination))
        new_data.append(stacked[name][combination])

    # Creating serie  
    reformatted = pd.Series(data=new_data, index=new_index, name=feat)

    # Return reformatted
    return reformatted