from pandas import DataFrame as DF
from pandas import read_csv , read_table
import numpy as np
import sys, string

def stop_err(msg):
    sys.stderr.write(msg)
    sys.exit()

infile = sys.argv[1]
outfile = sys.argv[2]

data=read_table(infile)
tdata=data.transpose()
tdata.to_csv(outfile)

