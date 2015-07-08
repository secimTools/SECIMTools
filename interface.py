#!/usr/bin/env python

# Built-in packages
import re

# Add-on packages
import pandas as pd


class wideToDesign:
    """ Class to handle generic data in a wide format with an associated design file. """
    def __init__(self, wide, design, uniqID, group=False, anno=False, clean_string=False):
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

            anno (list): A list of additional annotations that can be used to group
                items.

        Returns:
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

        # Import wide formatted data file
        try:
            self.uniqID = uniqID
            self.wide = pd.read_table(wide)
            if clean_string:
                self.wide[self.uniqID] = self.wide[self.uniqID].apply(lambda x: self._cleanStr(x))

            # Make sure index is a string and not numeric
            self.wide[self.uniqID] = self.wide[self.uniqID].astype(str)

            # Set index to uniqID column
            self.wide.set_index(self.uniqID, inplace=True)
        except:
            print "Please make sure that your data file has a column called '{0}'.".format(uniqID)
            raise ValueError

        # Import design file
        try:
            self.design = pd.read_table(design)

            # Make sure index is a string and not numeric
            self.design['sampleID'] = self.design['sampleID'].astype(str)
            self.design.set_index('sampleID', inplace=True)

            # Create a list of sampleIDs, but first check that they are present
            # in the wide data.
            self.sampleIDs = list()

            for sample in self.design.index.tolist():
                if sample in self.wide.columns:
                    self.sampleIDs.append(sample)

            # Drop design rows that are not in the wide data set
            self.design = self.design[self.design.index.isin(self.sampleIDs)]

        except:
            print "Please make sure that your design file has a column called 'sampleID'."
            raise ValueError

        # Save annotations
        self.anno = anno

        # Set up group information
        if group:
            if clean_string:
                self.group = self._cleanStr(group)
                self.design.columns = [self._cleanStr(x) for x in self.design.columns]
            else:
                self.group = group

            # combine group and anno
            if self.anno:
                keep = [self.group, ] + self.anno
            else:
                keep = [self.group, ]

            self.design = self.design[keep]   # Only keep group columns in the design file
            self.design[self.group] = self.design[self.group].astype(str)   # Make sure groups are strings

            # Create list of group levels
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

        Clean strings may need to be reverted back to original values for
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

            ID (str): A string referring to a uniqID in the dataset.

        Returns:
            (pd.DataFrame): with only the corresponding rows from the uniqID.

        """
        return self.wide[self.wide[self.uniqID] == ID]

    def keep_sample(self, sampleIDs):
        """ Keep only the given sampleIDs in the wide and design file.

        Arguments:
            :param list sampleIDs: A list of sampleIDs to keep.

        Returns:
            :rtype: wideToDesign
            :return: Updates the wideToDesign object to only have those sampleIDs.

        """
        self.sampleIDs = sampleIDs
        self.wide = self.wide[self.sampleIDs]
        self.design = self.design[self.design.index.isin(self.sampleIDs)]


class Flags:
    def __init__(self, index, column=''):
        """
        This class  creates an empty dataframe to hold the flag values for a dataset. The dataframe is created
        through instantiation and filled with 0's.

        Arguments:

            :param index: List of values to be used as the index of the data frame
            :type index: list

            :param column: Column name for the dataframe
            :type column: string

        """
        # Create DataFrame from index and columns
        self.df_flags = pd.DataFrame(index=index)
        if len(column) > 0:
            self.addColumn(column)

        # Set DF values equal to 0
        self.df_flags.fillna(0, inplace=True)

    def _testIfIndexesMatch(self, mask):
        """
        Before laying a mask over the Flags DataFrame, test if the mask's
        indexes match to avoid errors.

        :Arguments:
            :param mask: List of True and False values corresponding to flag
                         values.
            :type mask: list

        :Returns:
            :type boolean: True or false if the indexes match

        """

        result = self.df_flags.index.isin(mask.index).any()
        return result

    def update(self, mask, column=''):
        """
        Update the dataframe with 1's if the mask value is true

        :Arguments:
            :param mask: List of mask values. Must follow same structure as instantiated flag dataframe
            :type mask: list

            :param column: Column name to update in the flag frame. Not required
            :type column: String

        :Returns:
            Updated instance of the flag dataframe. The dataframe can be accessed through '.df_flags'.

        """

        # Update the values to 1's if they are true in the mask
        if len(column) > 0:
            if _testIfIndexesMatch(mask):
                self.df_flags.loc[mask.index, column] = mask.astype(int)
                #self.df_flags.loc[mask, column] = 1
        else:
            #self.df_flags[mask] = 1
            self.df_flags.loc[mask.index] = mask.astype(int)

    def addColumn(self, column, mask=[]):
        """
        Add a column to the flag DataFrame

        :Arguments:
            :param column: Name of the column to add to the DataFrame
            :type column: string | list of strings

            :param mask: List of True and False values corresponding to flag
                         values. OPTIONAL
            :type mask: list

        """
        self.df_flags[column] = 0

        # Update the column if a mask is given and the mask matches the index
        if len(mask) > 0:
            if _testIfIndexesMatch(mask):
             #self.update(mask=mask, column=column)
                self.df_flags.loc[mask.index, column] = mask.astype(int)

    def fillNa(self):
        """
        Fill the flag DataFrame with np.nan
        """

        # Fill the 0's with numpy.nan
        self.df_flags.replace(0, np.nan, inplace=True)

    @staticmethod
    def merge(flags):
        """
        Merge a list of DataFrames. This method will check to make sure all of the indices are the same for each
        DataFrame and will then return one merged DataFrame.

        Arguments:

            :param flags: List of DataFrames
            :type flags: list

        Returns:

            :return: DataFrame of merged flags
            :rtype: pandas.DataFrame

        """
        # Check the index of each dataframe before trying to merge
        counter = 0
        while counter < len(flags) - 1:
            if flags[counter].index.equals(flags[counter + 1].index):
                counter += 1
            else:
                print "Not all indexes are the same"
                raise SystemExit

        # Merge all flags together
        df_mergedFlags = pd.concat(flags, axis=1)

        # Return merged flag file
        return df_mergedFlags

if __name__ == '__main__':
    pass
