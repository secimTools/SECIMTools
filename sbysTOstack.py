from __future__ import print_function
from stack_sbys import sbysTOstack

sbysfilename=raw_input("Enter the name of side-by-side data file name\n")
ID=raw_input("Enter the name of the ID column (rows of the side-by-side data) e.g. 'probeset_id'\n")
prefix=raw_input("Enter the prefix of the names of the side-by-side columns to be stacked e.g. 'intensity_'\n")

sbysTOstack(sbysfilename,ID,prefix)
