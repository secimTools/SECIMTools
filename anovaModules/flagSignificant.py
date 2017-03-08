#Add-on packages
import numpy as np
import pandas as pd


def flagSignificant(fullRes):
    # Creatting a dataframe for flags
    flags_df = pd.DataFrame(index=fullRes.index)

    # Flagging lpval > 0.05
    flags_df["flag_significant_0.05_on"] = np.where(np.abs(fullRes["-log10_p-value_"])\
                                             > -np.log10(0.05),str(0),str(1));fullRes

    flags_df["flag_significant_0.01_on"] = np.where(np.abs(fullRes["-log10_p-value_"])\
                                             > -np.log10(0.01),str(0),str(1));fullRes

    flags_df["flag_significant_0.1_on"] = np.where(np.abs(fullRes["-log10_p-value_"])\
                                             > -np.log10(0.1),str(0),str(1));fullRes

    # Returning significant flags for that feature
    return flags_df