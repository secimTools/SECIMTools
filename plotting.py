#!/usr/bin/env python
""" Set of helper function and variables for plotting.

This module provides a set of functions and variables that will be useful for
plotting.

"""
import numpy as np
import pandas as pd


class ColorMarker:
    """ A class to handle colors for plotting with with DataFrames. """
    def __init__(self):
        """ Initialize the ColorMarker class.

        Attributes:
            :param list self._colors: A list of the standard python colors.

            :param list self._markers: A list of the standard python markers.

            :type self.colorMarker: list of tuple
            :param self.colorMarker: A list of tuples with (color, marker)
                pairs. There are 114 total combinations.

        """
        # A list of colors
        self._colors = ['k', 'b', 'g', 'c', 'm', 'y']

        # A list of markers
        self._markers = ['o', 's', '^', 'D', 'd', 'h', 'x', '*', '+', 'v', '<', '>', '1', '2', '3', '4', '8', 'p', 'H']

        # Create a list of color|marker pairs
        self.colorMarker = self._get_colors()

    def _get_colors(self):
        """ Get a set of color/marker combinations.

        :rtype: list of tuple
        :returns: A list of tuples containing color|marker pairs. There are a total
            of 114 combinations. Red and white are not used in this color scheme.
            Red is reserved for coloring points beyond a threshold, and white does not
            show up on white backgrounds.

        """
        comb = list()
        for marker in self._markers:
            for color in self._colors:
                comb.append((color, marker))

        return comb

    def groupColorMarker(self, design, groupID):
        """ Adds colors and markers to each group.

        Arguments:
            :type design: pandas.DataFrame
            :param design: A DataFrame with design information.

            :param str groupID: The column name containing group information in
                the design file.

        Returns:
            :rtype: tuple of dict
            :returns: A tuple of dictionaries with sampleID as the key and
                color or marker as value.

        """

        # Get group information from design
        grp = design.groupby(groupID)
        groupIDs = sorted(grp.groups.keys())
        groupNum = len(groupIDs)

        # Pull out the number of color|marker pairs needed for the number of groups
        if groupNum <= 114:
            colorMarker = self.colorMarker[:groupNum]
        else:
            print('Warning - you have more groups than color|marker combinations, will be repeating colors and markers!')
            multiplier = np.ceil(groupNum / 114.0)
            cm = self.colorMarker * multiplier
            colorMarker = cm[:groupNum]

        # Make a DataFrame of color|marker with groups
        dfColorMarker = pd.DataFrame(colorMarker, columns=['color', 'marker'], index=groupIDs)

        # Merge colors to design file
        merged = design.merge(dfColorMarker, how='left', left_on='group', right_index=True)

        # Generate color and marker dictionary
        myDict = merged.drop('group', axis=1).to_dict()
        colorsDict = myDict['color']
        markerDict = myDict['marker']

        return colorsDict, markerDict
