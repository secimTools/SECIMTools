#!/usr/bin/env python

# Built-in packages
import re

# Add-on packages
import pandas as pd


class wideToDesign:
    """ Class to handle generic data in a wide format with an associated design file. """
    def __init__(self, wide, design, uniqID, group=False, clean_string=False):
        """ Import and set-up data.

        Import data both wide formated data and a design file. Set-up basic
        attributes.

        Args:
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

            group (str): The name of column names in 'design' that give
                group information. For example: treatment

            clean_string (bool): If True remove special characters from strings
                in dataset.

        Returns:
            **Attribute**

            self.uniqID (str): The name of the unique identifier column in 'wide'
                (i.e. The column with compound/gene names).

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
        self.origString = dict()

        # Import wide formatted data file
        try:
            self.uniqID = uniqID
            self.wide = pd.read_table(wide)
            if clean_string:
                self.wide[self.uniqID] = self.wide[self.uniqID].apply(lambda x: self._cleanStr(x))
            self.wide.set_index(self.uniqID, inplace=True)
        except:
            print "Please make sure that your data file has a column called '{0}'.".format(uniqID)
            raise ValueError

        # Import design file
        try:
            self.design = pd.read_table(design, index_col='sampleID')

            # Create a list of sampleIDs, but first check that they are present
            # in the wide data.
            self.sampleIDs = list()

            for sample in self.design.index.tolist():
                if str(sample) in self.wide.columns:
                    self.sampleIDs.append(sample)
        except:
            print "Please make sure that your design file has a column called 'sampleID'."
            raise ValueError

        # Set up group information
        if group:
            if clean_string:
                self.group = self._cleanStr(group)
                self.design.columns = [self._cleanStr(x) for x in self.design.columns]
            else:
                self.group = group

            self.design = self.design[[self.group, ]]   # Only keep group columns in the design file
            grp = self.design.groupby(self.group)
            self.levels = sorted(grp.groups.keys())  # Get a list of group levels

    def _cleanStr(self, x):
        """ Clean strings so they behave.

        For some modules, uniqIDs and groups cannot contain spaces, '-', '*',
        '/', '+', or '()'. For example, statsmodel parses the strings and interprets
        them in the model.

        Args:
            x (str): A string that needs cleaning

        Returns:
            x (str): The cleaned string.

            self.origString (dict): A dictionary where the key is the new
                string and the value is the original string. This will be useful
                for reverting back to original values.

        """
        if isinstance(x, str):
            val = x
            x = x.replace(' ', '_')
            x = x.replace('-', '_')
            x = x.replace('*', '_')
            x = x.replace('/', '_')
            x = x.replace('+', '_')
            x = x.replace('(', '_')
            x = x.replace(')', '_')
            x = x.replace(')', '_')
            x = re.sub(r'^([0-9].*)', r'_\1', x)
            self.origString[x] = val
        return x

    def revertStr(self, x):
        """ Revert strings back to their original value so they behave well.

        Clean strings may need to be revertd back to original values for
        convience.

        Args:
            x (str): A string that needs cleaning

            self.origString (dict): A dictionary where the key is the cleaned
                string and the value is the original string.

        Returns:
            x (str): Original string.

        """
        if isinstance(x, str) and x in self.origString:
            x = self.origString[x]
        return x

    def melt(self):
        """ Convert a wide formated table to a long formated table.

        Args:
            self.wide (pd.DataFrame): A wide formatted table with compound/gene
                as row and sample as columns.

            self.uniqID (str): The name of the unique identifier column in 'wide'
                (i.e. The column with compound/gene names).

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

            self.uniqID (str): The name of the unique identifier column in 'wide'
                (i.e. The column with compound/gene names).

            ID (str): A string refering to a uniqID in the dataset.

        Returns:
            (pd.DataFrame): with only the corresponding rows from the uniqID.

        """
        return self.wide[self.wide[self.uniqID] == ID]

if __name__ == '__main__':
    pass
