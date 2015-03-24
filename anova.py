#!/usr/bin/env python

# Built-in packages
import argparse
from argparse import RawDescriptionHelpFormatter
import re
from itertools import combinations
from collections import defaultdict

# Add-on packages
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def getOptions():
    """ Function to pull in arguments """
    description = """ One-Way ANOVA """
    parser = argparse.ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--input", dest="fname", action='store', required=True, help="Input dataset in wide format.")
    parser.add_argument("--design", dest="dname", action='store', required=True, help="Design file.")
    parser.add_argument("--ID", dest="uniqID", action='store', required=True, help="Name of the column with unique identifiers.")
    parser.add_argument("--group", dest="group", action='store', nargs='*', required=True, help="Group/treatment identifiers. Can be multiple names separated by a space.")
    parser.add_argument("--std", dest="std", action='store', choices=['STD', 'MEAN'], required=False, help="Group/treatment identifiers. Can be multiple names separated by a space.")
    parser.add_argument("--out", dest="oname", action='store', required=True, help="Output file name.")
    parser.add_argument("--fig", dest="ofig", action='store', required=True, help="Output figure name for q-q plots [pdf].")
    parser.add_argument("--fig2", dest="ofig2", action='store', required=True, help="Output figure name for volcano plots [pdf].")
#     args = parser.parse_args()
    args = parser.parse_args(['--input', '/home/jfear/sandbox/secim/data/ST000015_AN000032_v2.txt',
                              '--design', '/home/jfear/sandbox/secim/data/ST000015_design_v2.tsv',
                              '--ID', 'Name',
                              '--group', 'treatment',
                              '--std', 'STD',
                              '--out', '/home/jfear/sandbox/secim/data/test.csv',
                              '--fig', '/home/jfear/sandbox/secim/data/test.pdf',
                              '--fig2', '/home/jfear/sandbox/secim/data/test2.pdf'])
    return(args)


class wideToDesign:
    """ Class to handle generic data in a wide format with an associated design file. """
    def __init__(self, wide, design, uniqID, group):
        """ Import and set-up data.

        Import data both wide formated data and a design file. Set-up basic
        attributes.

        Input:
            wide (TSV): A table in wide format with compounds/genes as rows and
                samples as columns.

                Name     sample1   sample2   sample3
                ------------------------------------
                one      10        20        10
                two      10        20        10

            design (TSV): A table relating samples ('sampleID') to groups or
                treatments.

                sampleID   group1  group2
                -------------------------
                sample1    g1      t1
                sample2    g1      t1
                sample3    g1      t1

            uniqID (str): The name of the unique identifier column in 'wide'
                (i.e. The column with compound/gene names).

        Returns:
            **Attribute**

            self.uniqID (np.array): An array of uniqIDs in self.wide.
                Typically this will be just a single string.

            self.wide (pd.DataFrame): A wide formatted table with compound/gene
                as row and sample as columns.

            self.sampleIDs (list): An list of sampleIDs. These will correspond
                to columns in self.wide.

            self.design (pd.DataFrame): A table relating sampleID to groups.

            self.group (list): A list of column names in self.design that give
                group information. For example: treatment, tissue

            self.levels (list): A list of levels in self.group. For example:
            trt1, tr2, control.

        """

        # Import data file to pandas
        try:
            self.uniqID = np.asarray(uniqID)
            self.wide = pd.read_table(wide)
            self.wide = self.wide.applymap(lambda x: self._cleanStr(x))     # Need to do this for the ols model
            self.wide.set_index(uniqID, inplace=True)
        except:
            print "Please make sure that your data file has a column called '{0}'.".format(uniqID)
            raise ValueError

        # Import design file to pandas
        try:
            self.design = pd.read_table(design, index_col='sampleID')

            # Set up group information
            self.sampleIDs = self.design.index.tolist()  # Create a list of sampleIDs
            self.group = list(group)
            self.design = self.design[self.group]   # Only keep group columns in the design file
            grp = self.design.groupby(self.group)
            self.levels = sorted(grp.groups.keys())  # Get a list of group levels
        except:
            print "Please make sure that your design file has a column called 'sampleID'."
            raise ValueError

    def _cleanStr(self, x):
        """ Clean strings so they behave well.

        uniqIDs cannot contain spaces, '-', or '()'. statsmodel parses the
        strings and interprets them in the model.

        Args:
            x (str): A string that needs cleaning

        Returns:
            x (str): The cleaned string.

        """
        if type(x) is str:
            x = x.replace(' ', '_')
            x = x.replace('-', '_')
            x = x.replace('(', '_')
            x = x.replace(')', '_')
            x = x.replace(')', '_')
            x = re.sub(r'^([0-9].*)', r'_\1', x)
        return x

    def melt(self):
        """ Convert a wide formated table to a long formated table.

        Args:
            self.wide (pd.DataFrame): A wide formatted table with compound/gene
                as row and sample as columns.

            self.uniqID (np.array): An array of uniqIDs in self.wide.
                Typically this will be just a single string.

            self.sampleIDs (list): An list of sampleIDs. These will correspond
                to columns in self.wide.

        Returns:
            **Attributes**

            self.long (pd.DataFrame): Creates a new attribute called self.long
                that also has group information merged to the dataset.

        """
        melted = pd.melt(self.wide.reset_index(), id_vars=self.uniqID, value_vars=self.sampleIDs,
                         var_name='sampleID')
        melted.set_index('sampleID', inplace=True)
        self.long = melted.join(self.design).reset_index()   # merge on group information using sampleIDs as key

    def transpose(self):
        """ Transpose the wide table and merge on treatment information.

        Args:
            self.wide (pd.DataFrame): A wide formatted table with compound/gene
                as row and sample as columns.

            self.design (pd.DataFrame): A table relating sampleID to groups.

        Returns:
            merged (pd.DataFrame): A wide formatted table with sampleID as row
                and compound/gene as column. Also has column with group ID.

        """
        trans = self.wide[self.sampleIDs].T

        # Merge on group information using table index (aka 'sampleID')
        merged = trans.join(self.design)
        merged.index.name = 'sampleID'
        return merged

    def getRow(self, ID):
        """ Get a row corresponding to a uniqID.

        Args:
            self.wide (pd.DataFrame): A wide formatted table with compound/gene
                as row and sample as columns.

            ID (str): A string refering to a uniqID in the dataset.

        Returns:
            (pd.DataFrame): with only the corresponding rows from the uniqID.

        """
        return self.wide[self.wide[self.uniqID] == ID]


def createCbn(dat):
    combo = defaultdict(dict)
    cbn = list(combinations(range(0, len(dat.levels)), 2))
    for c1, c2 in cbn:
        n1 = dat.levels[c1]
        n2 = dat.levels[c2]
        combo[(c1, c2)]['Name'] = '{0} - {1}'.format(n1, n2)
        combo[(c1, c2)]['Levels'] = [n1, n2]
        combo[(c1, c2)]['Diff'] = 'Diff of treatment = {0}'.format(combo[(c1, c2)]['Name'])
        combo[(c1, c2)]['StdErr'] = 'StdError for Diff of treatment = {0}'.format(combo[(c1, c2)]['Name'])
        combo[(c1, c2)]['Tval'] = 't Value for Diff of treatment = {0}'.format(combo[(c1, c2)]['Name'])
        combo[(c1, c2)]['pTval'] = 'Prob>|t| for Diff of treatment = {0}'.format(combo[(c1, c2)]['Name'])
        combo[(c1, c2)]['lpTval'] = '-log10(p-value) for Diff of treatment = {0}'.format(combo[(c1, c2)]['Name'])
        combo[(c1, c2)]['sTval'] = 'Sig Index for Diff of treatment = {0}'.format(combo[(c1, c2)]['Name'])
    return combo


def initResults(dat):
    """ Initialize the results dataset """
    base = ['GrandMean', 'ErrorSS']
    means = ['Mean treatment {0}'.format(x) for x in dat.levels]
    base2 = ['ErrorSS', 'ModelSS', 'TotalSS', 'NDF_treatment', 'DDF_treatment',
             'Variance_Residual', 'SampleVariance', 'RSquare', 'F_T3_treatment',
             'PrF_T3_treatment', 'MeanMean', 'StdDevMean']
    means2 = ['Std Mean treatment {0}'.format(x) for x in dat.levels]

    combo = createCbn(dat)
    diff = [combo[x]['Diff'] for x in sorted(combo.keys())]
    SE = [combo[x]['StdErr'] for x in sorted(combo.keys())]
    T = [combo[x]['Tval'] for x in sorted(combo.keys())]
    pT = [combo[x]['pTval'] for x in sorted(combo.keys())]
    lpT = [combo[x]['lpTval'] for x in sorted(combo.keys())]
    spT = [combo[x]['sTval'] for x in sorted(combo.keys())]

    cols = base + means + base2 + means2 + diff + SE + T + pT + lpT + spT
    return pd.DataFrame(index=dat.wide.index, columns=cols)


def sasSTD(dat, results):
    """ Function standardize using SAS STD procedure """
    # Get transposed table
    trans = dat.transpose()

    # group by treatment
    grp = trans.groupby(dat.group)

    # Calculate group means and std
    meanMean = grp.mean().mean()
    stdMean = grp.mean().std()

    # Standardize using sas STD method
    standard = grp.transform(lambda x: (x - meanMean[x.name]) / stdMean[x.name])
    merged = standard.join(dat.design)
    merged.index.name = 'sampleID'

    # Add results to output table
    sMeans = merged.groupby(dat.group).mean().T
    for compound in dat.wide.index.tolist():
        results.ix[compound, 'MeanMean'] = meanMean.ix[compound]
        results.ix[compound, 'StdDevMean'] = stdMean.ix[compound]
        for lvl in dat.levels:
            results.ix[compound, 'Std Mean treatment {0}'.format(lvl)] = sMeans.ix[compound, lvl]

    return merged


def sasMEAN(dat, results):
    """ Function standardize using SAS STD procedure """
    # Get transposed table
    trans = dat.transpose()

    # group by treatment
    grp = trans.groupby(dat.group)

    # Calculate group means and std
    meanMean = grp.mean().mean()

    # Standardize using sas MEAN method
    standard = grp.transform(lambda x: x - meanMean[x.name])
    merged = standard.join(dat.design)
    merged.index.name = 'sampleID'

    # Add results to output table
    sMeans = merged.groupby(dat.group).mean().T
    for compound in dat.wide.index.tolist():
        results.ix[compound, 'MeanMean'] = meanMean.ix[compound]
        results.ix[compound, 'StdDevMean'] = 1
        for lvl in dat.levels:
            results.ix[compound, 'Std Mean treatment {0}'.format(lvl)] = sMeans.ix[compound, lvl]

    return merged


def oneWay(dat, compound, results):
    """ One-way ANOVA """
    formula = '{0} ~ {1}'.format(str(compound), str(dat.group[0]))
    model_lm = ols(formula, data=dat.sdat).fit()
    results.ix[compound, 'F_T3_treatment'] = model_lm.fvalue
    results.ix[compound, 'PrF_T3_treatment'] = model_lm.f_pvalue
    results.ix[compound, 'ErrorSS'] = model_lm.ssr
    results.ix[compound, 'ModelSS'] = model_lm.ess
    results.ix[compound, 'TotalSS'] = model_lm.ess + model_lm.ssr
    results.ix[compound, 'NDF_treatment'] = model_lm.df_model
    results.ix[compound, 'DDF_treatment'] = model_lm.df_resid
    results.ix[compound, 'SampleVariance'] = dat.sdat[compound].var()
    results.ix[compound, 'RSquare'] = model_lm.rsquared
    return pd.Series(model_lm.resid, name=compound)


def calcDiff(dat, compound, grpMeans, combo, results):
    """ Calculate group means and the differences between group means """
    for lvl in dat.levels:
        results.ix[compound, 'Mean treatment {0}'.format(lvl)] = grpMeans.ix[compound, lvl]

    for key in sorted(combo.keys()):
        results.ix[compound, combo[key]['Diff']] = grpMeans.ix[compound, combo[key]['Levels'][0]] - grpMeans.ix[compound, combo[key]['Levels'][1]]


def calcDiffSE(dat, compound, combo, results):
    """ One-way ANOVA """
    # NOTE: This is a hack to get the right SEs
    # Run anova model using each treatment as the reference, pull out
    # corresponding SEs
    for i in range(0, len(dat.levels)):
        formula = '{0} ~ C({1}, Treatment(reference={2}))'.format(str(compound), str(dat.group[0]), i)
        model_lm = ols(formula, data=dat.sdat).fit()

        for j in range(i + 1, len(dat.levels)):
            SE = model_lm.bse[j]
            results.ix[compound, combo[(i, j)]['StdErr']] = SE


def tTest(compound, combo, results, cutoff=4):
    """ Calculate T-value and T-critical """
    for key in sorted(combo.keys()):
        results.ix[compound, combo[key]['Tval']] = results.ix[compound, combo[key]['Diff']] / results.ix[compound, combo[key]['StdErr']]
        results.ix[compound, combo[key]['pTval']] = stats.t.sf(abs(results.ix[compound, combo[key]['Tval']]), results.ix[compound, 'DDF_treatment']) * 2
        results.ix[compound, combo[key]['lpTval']] = -np.log10(results.ix[compound, combo[key]['pTval']])
        if np.abs(results.ix[compound, combo[key]['lpTval']]) > cutoff:
            results.ix[compound, combo[key]['sTval']] = 1
        else:
            results.ix[compound, combo[key]['sTval']] = 0


def qqPlot(resids, oname):
    """ Plot the residuals by sample """
    with PdfPages(oname) as pdf:
        trans = resids.T
        for col in trans.columns:
            fig = sm.graphics.qqplot(trans[col], fit=True, line='q')
            fig.suptitle(col)
            pdf.savefig(fig)
            plt.close(fig)


def volcano(combo, results, oname, cutoff=4):
    """ Plot volcano plots """
    with PdfPages(oname) as pdf:
        for key in sorted(combo.keys()):
            diff = combo[key]['Diff']
            pval = combo[key]['lpTval']

            # Set up figure
            fig = plt.figure(figsize=(8, 8))
            fig.suptitle(combo[key]['Name'])
            ax = fig.add_subplot(111)

            # Plot all results
            results.plot(x=diff, y=pval, kind='scatter', ax=ax)

            # Color results beyond threshold red
            subset = results[results[pval] > cutoff]
            if not subset.empty:
                subset.plot(x=diff, y=pval, kind='scatter', color='r', ax=ax)

            plt.axhline(cutoff, color='r', ls='--', lw=2)
            pdf.savefig(fig)
            plt.close(fig)


def cleanCol(col, decimals=4):
    """ Round floats to `decimals` places """
    try:
        col = np.round(col, decimals=decimals)
    except:
        pass
    return col


def main():
    # Command line options
    args = getOptions()

    # Import data
    dat = wideToDesign(args.fname, args.dname, args.uniqID, args.group)
    results = initResults(dat)

    # Standardize the data
    try:
        if args.std == 'STD':
            dat.sdat = sasSTD(dat, results)
        elif args.sstd == 'MEAN':
            dat.sdat = sasMEAN(dat, results)
    except:
        pass

    # NOTE: JMP does not seem to use the standardized values. The output looks
    # identical with and without standardization. Set sdat back to
    # unstandardized results.
    dat.sdat = dat.transpose()

    # Group by Treatment
    grp = dat.sdat.groupby(dat.group)
    grpMeans = grp.mean().T
    combo = createCbn(dat)

    resids = list()
    # Iterate over compound
    for compound in dat.wide.index.tolist():
        # Get Overall Mean
        results.ix[compound, 'GrandMean'] = dat.sdat[compound].mean()
        # run one-way anova
        resid = oneWay(dat, compound, results)
        resids.append(resid)

        # Calculate mean differences
        calcDiff(dat, compound, grpMeans, combo, results)

        # Calculate SE of difference between means
        calcDiffSE(dat, compound, combo, results)

        # Calculate T-test
        tTest(compound, combo, results)

    residDat = pd.concat(resids, axis=1)

    # Generate qqplots
    qqPlot(residDat, args.ofig)

    # Generate Volcano plots
    volcano(combo, results, args.ofig2)

    # write results table
    clean = results.apply(lambda x: cleanCol(x))
    clean.to_csv(args.oname, sep="\t")


if __name__ == '__main__':
    main()
