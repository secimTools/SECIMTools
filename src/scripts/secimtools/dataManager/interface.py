#!/usr/bin/env python
"""
Secim Tools data interface library.
"""
# Built-in packages
import re
import sys

# Add-on packages
import numpy as np
import pandas as pd


class wideToDesign:
    """ Class to handle generic data in a wide format with an associated design file. """
    def __init__(self, wide, design, uniqID, group=False, runOrder=False, anno=False, clean_string=True,
                 infer_sampleID=True, keepSample=True, logger=None):
        """ Import and set-up data.

        Import data both wide formated data and a design file. Set-up basic
        attributes.

        :Arguments:
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

            infer_sampleID (bool): If True infer "sampleID" from different capitalizations.

            anno (list): A list of additional annotations that can be used to group
                items.

        :Returns:
            **Attribute**

            self.uniqID (str): The name of the unique identifier column in 'wide'
                (i.e. The column with compound/gene names).

            self.wide (pd.DataFrame): A wide formatted table with compound/gene
                as row and sample as columns.

            self.sampleIDs (list): A list of sampleIDs. These will correspond
                to columns in self.wide.

            self.design (pd.DataFrame): A table relating sampleID to groups.

            self.group (list): A list of column names in self.design that give
                group information. For example: treatment, tissue

            anno (list): A list of additional annotations that can be used to group
                items.

            self.levels (list): A list of levels in self.group. For example:
                trt1, tr2, control.

        """
        # Setting logger
        if logger is None:
            self.logger = False
        else:
            self.logger = logger

        # Saving original str
        self.origString = dict()

        # Import wide formatted data file
        try:
            self.uniqID = uniqID
            self.wide = pd.read_table(wide)
            if clean_string:
                self.wide[self.uniqID] = self.wide[self.uniqID].apply(lambda x: self._cleanStr(str(x)))
                self.wide.rename(columns=lambda x: self._cleanStr(x), inplace=True)

            # Make sure index is a string and not numeric
            self.wide[self.uniqID] = self.wide[self.uniqID].astype(str)

            # Set index to uniqID column
            self.wide.set_index(self.uniqID, inplace=True)

        except ValueError:
            if self.logger:
                self.logger.error("Please make sure that your data file has a column called '{0}'.".format(uniqID))
            else:
                print(("Please make sure that your data file has a column called '{0}'.".format(uniqID)))
            raise ValueError

        # Import design file
        try:
            self.design = pd.read_table(design)

            # This part of the script allows the user to use any capitalization of "sampleID"
            # ie. "sample Id" would be converted to "sampleID".
            # If you want to accept only the exact capitalization turn infer_sampleID to Fake
            ## AMM added additional backslash to \s in regex below
            if infer_sampleID:
                renamed = {column: re.sub(r"[s|S][a|A][m|M][p|P][l|L][e|E][\\s?|_?][I|i][d|D]",
                                          "sampleID", column) for column in self.design.columns}
                self.design.rename(columns=renamed, inplace=True)
                log_msg = "Inferring 'sampleID' from data. This will accept different capitalizations of the word"
                if self.logger:
                    self.logger.info(log_msg)
                else:
                    print(log_msg)

            # Make sure index is a string and not numeric
            self.design['sampleID'] = self.design['sampleID'].astype(str)
            self.design.set_index('sampleID', inplace=True)
            #print(self.design)

            # Cleaning design file
            if clean_string:
                self.design.rename(index=lambda x: self._cleanStr(x), inplace=True)

            # Create a list of sampleIDs, but first check that they are present
            # in the wide data.
            self.sampleIDs = list()

            for sample in self.design.index.tolist():
                if sample in self.wide.columns:
                    self.sampleIDs.append(sample)
                else:
                    if self.logger:
                        self.logger.warn("Sample {0} missing in wide dataset".format(sample))
                    else:
                        print(("WARNING - Sample {0} missing in wide dataset".format(sample)))

            for sample in self.wide.columns.tolist():
                if not (sample in self.design.index):
                    if keepSample:
                        if self.logger:
                            self.logger.warn("Sample {0} missing in design file".format(sample))
                        else:
                            print(("WARNING - Sample {0} missing in design file".format(sample)))
                    else:
                        if self.logger:
                            self.logger.error("Sample {0} missing in design file".format(sample))
                            raise
                        else:
                            print(("ERROR - Sample {0} missing in design file".format(sample)))
                            raise

            # Drop design rows that are not in the wide data set
            self.design = self.design[self.design.index.isin(self.sampleIDs)]
            #print("DEBUG: design")
            #print(self.design)
            # Removing characters from data!!!!!!(EXPERIMENTAL)
            self.wide.replace(r'\D', np.nan, regex=True, inplace=True)
        # Possible bad design, bare except should not be used
        except SystemError:
            print(("Error:", sys.exc_info()[0]))
            raise

        # Save annotations
        self.anno = anno

        # Save runOrder
        self.runOrder = runOrder

        # Set up group information
        if group:
            if clean_string:
                self.group = self._cleanStr(group)
                self.design.columns = [self._cleanStr(x) for x in self.design.columns]
            else:
                self.group = group
            keep = self.group.split(",")
            # combine group, anno and runorder
            if self.runOrder and self.anno:
                keep = keep + [self.runOrder, ] + self.anno
            elif self.runOrder and not self.anno:
                keep = keep + [self.runOrder, ]
            elif not self.runOrder and self.anno:
                keep = keep + self.anno
            # Check if groups, runOrder and levels columns exist in the design file
            designCols = self.design.columns.tolist()
            if keep == designCols:
            # Check if columns exist on design file.
                self.design = self.design[keep]   # Only keep group columns in the design file
                self.design[self.group] = self.design[self.group].astype(str)   # Make sure groups are strings
            # Create list of group levels
            grp = self.design.groupby(self.group)
            self.levels = sorted(grp.groups.keys())  # Get a list of group levels
        else:
            self.group = None

        # Keep samples listed in design file
        if keepSample:
            self.keep_sample(self.sampleIDs)

    def _cleanStr(self, x):
        """ Clean strings so they behave.

        For some modules, uniqIDs and groups cannot contain spaces, '-', '*',
        '/', '+', or '()'. For example, statsmodel parses the strings and interprets
        them in the model.

        :Arguments:
            x (str): A string that needs cleaning

        :Returns:
            x (str): The cleaned string.

            self.origString (dict): A dictionary where the key is the new
                string and the value is the original string. This will be useful
                for reverting back to original values.

        """
        if isinstance(x, str):
            val = x
            x = re.sub(r'^-([0-9].*)', r'__\1', x)
            x = x.replace(' ', '_')
            x = x.replace('.', '_')
            x = x.replace('-', '_')
            x = x.replace('*', '_')
            x = x.replace('/', '_')
            x = x.replace('+', '_')
            x = x.replace('(', '_')
            x = x.replace(')', '_')
            x = x.replace('[', '_')
            x = x.replace(']', '_')
            x = x.replace('{', '_')
            x = x.replace('}', '_')
            x = x.replace('"', '_')
            x = x.replace('\'', '_')
            x = re.sub(r'^([0-9].*)', r'_\1', x)
            self.origString[x] = val
        return x

    def revertStr(self, x):
        """ Revert strings back to their original value so they behave well.

        Clean strings may need to be reverted back to original values for
        convience.

        :Arguments:
            x (str): A string that needs cleaning

            self.origString (dict): A dictionary where the key is the cleaned
                string and the value is the original string.

        :Returns:
            x (str): Original string.

        """
        if isinstance(x, str) and x in self.origString:
            x = self.origString[x]
        return x

    def melt(self):
        """ Convert a wide formated table to a long formated table.

        :Arguments:
            self.wide (pd.DataFrame): A wide formatted table with compound/gene
                as row and sample as columns.

            self.uniqID (str): The name of the unique identifier column in 'wide'
                (i.e. The column with compound/gene names).

            self.sampleIDs (list): An list of sampleIDs. These will correspond
                to columns in self.wide.

        :Returns:
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

        :Arguments:
            self.wide (pd.DataFrame): A wide formatted table with compound/gene
                as row and sample as columns.

            self.design (pd.DataFrame): A table relating sampleID to groups.

        :Returns:
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

        :Arguments:
            self.wide (pd.DataFrame): A wide formatted table with compound/gene
                as row and sample as columns.

            self.uniqID (str): The name of the unique identifier column in 'wide'
                (i.e. The column with compound/gene names).

            ID (str): A string referring to a uniqID in the dataset.

        :Returns:
            (pd.DataFrame): with only the corresponding rows from the uniqID.

        """
        return self.wide[self.wide[self.uniqID] == ID]

    def keep_sample(self, sampleIDs):
        """
        Keep only the given sampleIDs in the wide and design file.

        :Arguments:
            :param list sampleIDs: A list of sampleIDs to keep.

        :Returns:
            :rtype: wideToDesign
            :return: Updates the wideToDesign object to only have those sampleIDs.

        """
        self.sampleIDs = sampleIDs
        self.wide = self.wide[self.sampleIDs]
        self.design = self.design[self.design.index.isin(self.sampleIDs)]

    def removeSingle(self):
        """
        Removes groups with just one sample
        """
        if self.group:
            for level, current in self.design.groupby(self.group):
                if len(current) < 2:
                    self.design.drop(current.index, inplace=True)
                    self.wide.drop(current.index, axis=1, inplace=True)
                    log_msg = """Your group '{0}' has only one element,"
                                 "this group is going to be removed from"
                                 "further calculations.""".format(level)
                    if self.logger:
                        self.logger.warn(log_msg)
                    else:
                        print(log_msg)

    def dropMissing(self):
        """
        Drops rows with missing data
        """
        # Asks if any missing value
        if np.isnan(self.wide.values).any():
            # Count original number of rows
            n_rows = len(self.wide.index)

            # Drop missing values
            self.wide.dropna(inplace=True)

            # Count the dropped rows
            n_rows_kept = len(self.wide.index)

            # Logging!!!
            log_msg = """Missing values were found in wide data.
                         [{0}] rows were dropped""".format(n_rows - n_rows_kept)
            if self.logger:
                self.logger.warn(log_msg)
            else:
                print(log_msg)


class annoFormat:
    """ Class to handle generic data in a wide format with an associated design file. """
    def __init__(self, data, uniqID, mz, rt, anno=False, clean_string=True):
        """ Import and set-up data.

        Import data both wide formated data and a design file. Set-up basic
        attributes.

        :Arguments:
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

            anno (list): A list of additional annotations that can be used to group
                items.

        :Returns:
            **Attribute**

            self.uniqID (str): The name of the unique identifier column in 'wide'
                (i.e. The column with compound/gene names).

            self.wide (pd.DataFrame): A wide formatted table with compound/gene
                as row and sample as columns.

            self.sampleIDs (list): A list of sampleIDs. These will correspond
                to columns in self.wide.

            self.design (pd.DataFrame): A table relating sampleID to groups.

            self.group (list): A list of column names in self.design that give
                group information. For example: treatment, tissue

            anno (list): A list of additional annotations that can be used to group
                items.

            self.levels (list): A list of levels in self.group. For example:
                trt1, tr2, control.

        """
        self.origString = dict()

        # Import anno formatted data file
        try:
            self.uniqID = uniqID
            self.mz = mz
            self.rt = rt

            # Trying to import
            self.data = pd.read_table(data)

            if clean_string:
                self.data[self.uniqID] = self.data[self.uniqID].apply(lambda x: self._cleanStr(x))
                self.data.rename(columns=lambda x: self._cleanStr(x), inplace=True)

            # Make sure index is a string and not numeric
            self.data[self.uniqID] = self.data[self.uniqID].astype(str)

            # Set index to uniqID column
            self.data.set_index(self.uniqID, inplace=True)

            # If not annotation then ignoring additional columns
            self.anno = None
            if not(anno):
                self.data = self.data[[self.mz, self.rt]]
            else:
                self.anno = self.data.columns.tolist()
                self.anno.remove(self.mz)
                self.anno.remove(self.rt)
        except ValueError:
            print(("Data file must have columns called '{0}','{1}' and '{2}'.".format(uniqID, mz, rt)))
            raise ValueError

    def _cleanStr(self, x):
        """ Clean strings so they behave.

        For some modules, uniqIDs and groups cannot contain spaces, '-', '*',
        '/', '+', or '()'. For example, statsmodel parses the strings and interprets
        them in the model.

        :Arguments:
            x (str): A string that needs cleaning

        :Returns:
            x (str): The cleaned string.

            self.origString (dict): A dictionary where the key is the new
                string and the value is the original string. This will be useful
                for reverting back to original values.

        """
        if isinstance(x, str):
            val = x
            x = x.replace(' ', '_')
            x = x.replace('.', '_')
            x = x.replace('-', '_')
            x = x.replace('*', '_')
            x = x.replace('/', '_')
            x = x.replace('+', '_')
            x = x.replace('(', '_')
            x = x.replace(')', '_')
            x = x.replace('[', '_')
            x = x.replace(']', '_')
            x = x.replace('{', '_')
            x = x.replace('}', '_')
            x = x.replace('"', '_')
            x = x.replace('\'', '_')
            x = re.sub(r'^([0-9].*)', r'_\1', x)
            self.origString[x] = val
        return x


if __name__ == '__main__':
    pass
