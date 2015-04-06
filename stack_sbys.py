from __future__ import print_function
from pandas import DataFrame as DF
from pandas import read_csv 
import numpy as np


def sbysTOstack(sbysfilename,ID,prefix):

    data=read_csv(sbysfilename)
    data=data.set_index(ID)
    columns_to_be_stacked=[]
    for i, name in enumerate(data.columns):
        if name.startswith(prefix):
            columns_to_be_stacked.append(name)
    data=DF(data, columns=columns_to_be_stacked)
    stackDF=data.T.stack().T
    stack=DF({prefix:stackDF})
    stack.to_csv("stack"+sbysfilename)

