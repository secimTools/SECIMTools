#Add-on packages
import numpy as np
import pandas as pd


def flagSignificant(fullRes):
    """
    This fuction flags all the significant values for 3 pvalues: 0.1, 
    0.01 and 0.05.

    :Arguments:
        :type fullRes: pd.DataFrame
        :param fullRes: dataFrame with all the results for the model

    :Returns:
        :rtype df_flags: pd.DataFrame.
        :return df_flags: dataFrame containing flags for the 3 p-values. 

    """
    # Creatting a dataframe for flags
    df_flags = pd.DataFrame(index=fullRes.index)

    # Flagging lpval > 0.05
    df_flags["flag_significant_0.05_on"] = np.where(np.abs(fullRes["-log10_p-value_"]) > -np.log10(0.05),str(1),str(0));fullRes
    df_flags["flag_significant_0.01_on"] = np.where(np.abs(fullRes["-log10_p-value_"]) > -np.log10(0.01),str(1),str(0));fullRes
    df_flags["flag_significant_0.1_on"]  = np.where(np.abs(fullRes["-log10_p-value_"]) > -np.log10(0.1), str(1),str(0));fullRes
    
    # Returning significant flags for that feature
    return df_flags
