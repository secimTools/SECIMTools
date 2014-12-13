#OS Specific Modules
import getopt
import sys, os

#Data Manipulation
import pandas as pnds
import string

#Numerics
import numpy as np
from scipy import stats

if len(sys.argv[1:])<=1:  ### Indicates that there are insufficient number of command-line arguments
    print "Warning! Please designate a tab-delimited input expression file in the command-line"
    print "Example: anova.py --infile inputfile.dat --group-column group_column --groups group1,group2,group3 --test-column test_column"
    sys.exit()
else:
    options, remainder = getopt.getopt(sys.argv[1:],'', ['infile=','group-column=','groups=','test-column='])
    for opt, arg in options:
        if opt == '--infile': in_file=arg
        elif opt == '--group-column': group_column=arg
        elif opt == '--groups': groups=arg
        elif opt == '--test-column': test_column=arg
        else:
            print "Warning! Command-line argument: %s not recognized. Exiting..." % opt; sys.exit()

#Define pandas data frame
#must specify that blank space " " is NaN  
df = pnds.read_table(in_file, na_values=[" "])

#Define groups for which anova will be run
groups = groups.split(',')

#Tell user what arguments are being used for the ANOVA
print "Running ANOVA on file: "
print in_file, "\n"
print "Test column: "
print test_column, "\n"
print "Sample groups are: "
print '\t'.join(groups), "\n"

data = {}
for i in groups:
    data[i] = df[df[group_column] == i][test_column]

f_val, p_val = stats.f_oneway(*data.values())
print "One-way ANOVA:"
print "P = ", p_val
print "F = ", f_val



