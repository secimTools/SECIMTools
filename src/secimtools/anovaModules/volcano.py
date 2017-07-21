#Add-on packages
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Plotting packages
from secimtools.visualManager import module_box as box
from secimtools.visualManager import module_hist as hist
from secimtools.visualManager import module_lines as lines
from secimtools.visualManager import module_scatter as scatter
from secimtools.visualManager.manager_color import colorHandler
from secimtools.visualManager.manager_figure import figureHandler

def volcano(combo, results, oname, cutoff=2):
    """ 
    Plot volcano plots.

    Creates volcano plots to compare means, for all pairwise differences.

    :Arguments:

        :type combo: dictionary
        :param combo: A dictionary of dictionaries with all possible pairwise
            combinations. Used this to create the various column headers in the
            results table.

        :type results: pandas DataFrame
        :param results: TODO

        :type oname: string
        :param oname: Name of the output file in pdf format.
       
        :type cutoff: int
        :param cutoff: The cutoff value for significance.

    :Returns:
        :rtype: PD
        :returns: Outputs a pdf file containing all plots.

    """
    # Getting data for lpvals
    lpvals = {col.split("_")[-1]:results[col] for col in results.columns.tolist() \
            if col.startswith("-log10_p-value_")}

    # Gettign data for diffs
    difs   = {col.split("_")[-1]:results[col] for col in results.columns.tolist() \
            if col.startswith("diff_of")}

    # Making plots
    with PdfPages(oname) as pdf:
        for key in sorted(difs.keys()):
            # Set Up Figure
            volcanoPlot = figureHandler(proj="2d")

            # Plot all results
            scatter.scatter2D(x=list(difs[key]), y=list(lpvals[key]), 
                                colorList=list('b'), ax=volcanoPlot.ax[0])

            # Color results beyond treshold red
            cutLpvals = lpvals[key][lpvals[key]>cutoff]
            if not cutLpvals.empty:
                cutDiff = difs[key][cutLpvals.index]
                scatter.scatter2D(x=list(cutDiff), y=list(cutLpvals), 
                                colorList=list('r'), ax=volcanoPlot.ax[0])

            # Drawing cutoffs
            lines.drawCutoffHoriz(y=cutoff, ax=volcanoPlot.ax[0])

            # Format axis (volcanoPlot)
            volcanoPlot.formatAxis(axTitle=key, grid=False,
                yTitle="-log10(p-value) for Diff of treatment = {0}".format(key),
                xTitle="Diff of treatment = {0}".format(key))

            # Add figure to PDF
            volcanoPlot.addToPdf(pdfPages=pdf)
