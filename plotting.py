#!/usr/bin/env python
""" Set of helper function and variables for plotting.

This module provides a set of functions and variables that will be useful for
plotting.

"""


class ColorMarker:
    def __init__(self):
        # A list of colors
        self._colors = ['k', 'b', 'g', 'c', 'm', 'y']
        # A list of markers
        self._markers = ['o', 's', '^', 'D', 'd', 'h', 'x', '*', '+', 'v', '<', '>', '1', '2', '3', '4', '8', 'p', 'H']

    def get_colors(self):
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
