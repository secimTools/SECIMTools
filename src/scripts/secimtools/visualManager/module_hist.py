######################################################################################
#
# Module: module_hist.py
#
# AUTHOR: Matt Thoburn <mthoburn@ufl.edu>
#
# DESCRIPTION: This module contains methods to plot a histogram in matplotlib
#
#######################################################################################

import matplotlib.pyplot as plt
import pandas as pd


def quickHist(ax, dat, orientation="vertical", color="#0000FF"):
    """
    draws a histogram with minimal formatting

    :Arguments:
        :type ax: matplotlib axis
        :param ax: axis to plot data on

        :type dat: pandas dataframe
        :param dat: data to be plotted

        :type orientation: string
        :param orientation: vertical or horizontal
    """
    dat.hist(ax=ax, orientation=orientation, color=color)
    return ax


def serHist(
    ax, dat, color, range=None, bins=10, normed=False, orientation="vertical", label=""
):
    """
    draws a histogram with formatting options and coloring for groups

    :Arguments:
        :type ax: matplotlib axis
        :param ax: axis to plot data on

        :type dat: pandas Series
        :param dat: data to be plotted

        :type colors: list of colors
        :param colors: colors for groups

        :type range: tuple of numbers or None
        :param range: lower and upper range of bins

        :type bins: int
        :param bins: number of bins for histogram

        :type normed: boolean or None
        :param normed: normalize integral of histogram to 1 if true

        :type orientation: string
        :param orientation: vertical or horizontal
    """
    ax.hist(
        dat,
        color=color,
        alpha=0.7,
        orientation=orientation,
        range=range,
        bins=bins,
        label=label,
    )


def hist(
    ax, dat, colors, range=None, bins=10, normed=False, orientation="vertical", label=""
):
    """
    draws a histogram with formatting options and coloring for groups

    :Arguments:
        :type ax: matplotlib axis
        :param ax: axis to plot data on

        :type dat: pandas dataframeO
        :param dat: data to be plotted

        :type colors: list of colors
        :param colors: colors for groups

        :type range: tuple of numbers or None
        :param range: lower and upper range of bins

        :type bins: int
        :param bins: number of bins for histogram

        :type normed: boolean or None
        :param normed: normalize integral of histogram to 1 if true

        :type orientation: string
        :param orientation: vertical or horizontal
    """
    headers = dat.wide.columns.values
    groupList = dat.design[dat.group].values

    for group, color in zip(dat.levels, colors):
        valPerGroup = dat.wide[dat.design.index[dat.design[dat.group] == group]].values
        lists = sum(list(map(list, valPerGroup)), [])
        ax.hist(
            lists,
            color=color,
            alpha=0.7,
            orientation=orientation,
            range=range,
            bins=bins,
            normed=normed,
            label=label,
        )
