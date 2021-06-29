#!/usr/bin/env python

# Built-in packages
import re
import sys

# Add-on packages
import pandas as pd

class Flags:
    def __init__(self, index):
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

        # Create a list to store column names
        self._columns = list()

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
            if self._testIfIndexesMatch(mask):
                self.df_flags.loc[mask.index, column] = mask.astype(int)
        else:
            self.df_flags.loc[mask.index, self._columns] = mask.astype(int)

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

        # Store column names
        if isinstance(column, str):
            self._columns.append(column)
        else:
            self._columns.extend(column)

        # Update the column if a mask is given and the mask matches the index
        if len(mask) > 0:
            self.update(mask=mask, column=column)

    def fillNa(self):
        """
        Fill the flag DataFrame with np.nan
        """

        # Fill the 0's with numpy.nan
        self.df_flags.replace(0, np.nan, inplace=True)

    def testOverlap(self, indices):
        """ Test if a list of indeces overlap. """

        # TODO: Trying to figure out the best algorithm to test if indeces are
        # the sam.
        for i, index in enumerate(indices):
            if i == 0:
                overlap = set(index)
            else:
                if overlap.intersection(set(index)):
                    overlap = overlap.union(set(index))

    def splitFlags(self):
        """ Split large DataFrame into individual DataFrames per column

        :Returns:
            :rtype: dictionary
            :returns: Dictionary of pandas.DataFrame

        """
        # List to hold dataframes
        df_list= []

        # Loop through columns and build a dataframe
        for column in self._columns:
            exec('df_' + str(column) + '= pd.DataFrame(' +
                    'data=self.df_flags[column], index=self.df_flags.index,' +
                    'columns=[column])')
            # Add newly created DataFrame to df dictionary
            exec('df_dict.append(df_' + str(column) + ')')

        # Return df_dict
        return df_list

    @staticmethod
    def _mergeIndex(indices):
        """ Function to check for overlap for a list of indices.

        This function is based on:
        http://stackoverflow.com/questions/9110837/python-simple-list-merging-based-on-intersections

        :param list indices: A list of pd.Index

        """
        # Convert index into set
        sets = [set(ind) for ind in indices if len(ind)]
        merged = 1
        while merged:
            merged = 0
            results = []
            while sets:
                common, rest = sets[0], sets[1:]
                sets = []
                for x in rest:
                    if x.isdisjoint(common):
                        # If they don't overlap then append
                        sets.append(x)
                    else:
                        # If they overlap, take the union
                        merged = 1
                        common |= x
                results.append(common)
            sets = results
        return sets

    @staticmethod
    def merge(flags):
        """
        Merge a list of DataFrames. This method will check to make sure all of the indices are the same for each
        DataFrame and will then return one merged DataFrame.

        :Arguments:

            :param flags: List of DataFrames
            :type flags: list

        :Returns:

            :return: DataFrame of merged flags
            :rtype: pandas.DataFrame

        """
        # Check the index of each dataframe before trying to merge
        mergeIndex = Flags._mergeIndex([x.index for x in flags])

        if len(mergeIndex) == 1:
            # Merge all flags together
            # NOTE: Pandas cannot store NAN values as a int. If there are NAN
            # then the column is converted to a float.
            df_mergedFlags = pd.concat(flags, axis=1)

            # Return merged flag file
            return df_mergedFlags
        else:
            print("Not all indexes overlap. Check that flags are features OR \
                   samples.")
            raise SystemExit


if __name__ == '__main__':
    pass
