#!/usr/bin/env python
################################################################################
# DATE: 2017/01/17
#
# MODULE: modulated_modularity_clustering.py
#
# VERSION: 1.2
#
# ADDAPTED: Miguel Ibarra (miguelib@ufl.edu) Matt Thoburn (mthoburn@ufl.edu) 
#
# DESCRIPTION: This tool runs modulated modularity clustering (mmc)
################################################################################
# Import built-in libraries
import os
import csv
import logging
import argparse
from contextlib import contextmanager
# Import add-on libraries
import scipy
import numpy as np
import pandas as pd
import decimal
from numpy.testing import assert_allclose
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
# Import local data libraries
from secimtools.dataManager import logger as sl
from secimtools.dataManager.interface import wideToDesign
# Import local plotting libraries
from secimtools.visualManager import module_heatmap as hm
from secimtools.visualManager.manager_color import colorHandler
from secimtools.visualManager.manager_figure import figureHandler
from secimtools.visualManager.module_mmc import expansion, get_clustering


def getOptions(myOpts = None):
    # Get command line arguments.
    parser = argparse.ArgumentParser()
    # Standard Input
    standard = parser.add_argument_group(title='Standard input', 
                                description='Standard input for SECIM tools.')
    standard.add_argument( "-i","--input", dest="input", action='store', 
                        required=True, help="Input dataset in wide format.")
    standard.add_argument("-d" ,"--design",dest="design", action='store', 
                        required=True, help="Design file.")
    standard.add_argument("-id", "--ID",dest="uniqID", action='store', 
                        required=True, help="Nam of the column with unique"\
                        " identifiers.")
    # Tool Input
    tool = parser.add_argument_group(title='Tool input', 
                                description='Specific input for the tools.')
    tool.add_argument("-c",'--correlation', dest="correlation", default="pearson",
                        choices=("pearson", "kendall", "spearman"),
                        help=("Compute correlation coefficients using either "
                        "'pearson' (standard correlation coefficient), "
                        "'kendall' (Kendall Tau correlation coefficient), or "
                        "'spearman' (Spearman rank correlation)."))
    tool.add_argument("-sl",'--sigmaLow',dest="sigmaLow", type=float, 
                        default=0.05, help="Low value of sigma (Default: 0.05).")
    tool.add_argument("-sh",'--sigmaHigh',dest="sigmaHigh", type=float, 
                        default=0.50, help="High value of sigma (Default: 0.50).")
    tool.add_argument("-sn",'--sigmaNum',dest="sigmaNum", type=int, 
                        default=451, help="Number of values of sigma to search"
                        " (Default: 451).")
    # Tool Output
    output = parser.add_argument_group(title='Output paths', 
                        description='Output paths for the tools.')
    output.add_argument('-f', "--figure", dest="figure", required=True,
                        help="MMC Heatmaps")
    output.add_argument('-o',"--out", dest="out", required=True, 
                        help="Output TSV name")
    # Plot Options
    plot = parser.add_argument_group(title='Plot options')
    plot.add_argument("-pal","--palette",dest="palette",action='store',required=False, 
                        default="diverging", help="Name of the palette to use.")
    plot.add_argument("-col","--color",dest="color",action="store",required=False, 
                        default="Spectral_10", help="Name of a valid color scheme"\
                        " on the selected palette")

    args = parser.parse_args()

    # Standardize paths
    args.out    = os.path.abspath(args.out)
    args.input  = os.path.abspath(args.input)
    args.design = os.path.abspath(args.design)
    args.figure = os.path.abspath(args.figure)

    return args


def nontechnical_analysis(args, df, mask, C, clustering):
    # Re-order things more palatably for the user,
    # based on the results of the technical analysis.

    # Get the map from the name to the original row index.
    all_row_names = df.index.values
    row_index_map = {s : i for i, s in enumerate(all_row_names)}

    # If some variables are uninformative for clustering,
    # the correlation matrix and the cluster vector will have smaller
    # dimensions than the number of rows in the original data frame.
    remaining_row_names = df[mask].index.values

    # Count the variables included in the clustering.
    p = clustering.shape[0]

    # Count the clusters.
    k = clustering.max() + 1

    # To sort the modules and to sort the variables within the modules,
    # we want to use absolute values of correlations.
    C_abs = np.abs(C)

    # For each cluster, get its indices and its submatrix of C_abs.
    selections = []
    submatrices = []
    degrees = np.zeros(p, dtype=float)
    for i in range(k):
        selection = np.flatnonzero(clustering == i)
        selections.append(selection)
        submatrix = C_abs[np.ix_(selection, selection)]
        submatrices.append(submatrix)
        if selection.size > 1:
            denom = selection.size - 1
            degrees[selection] = (submatrix.sum(axis=0) - 1) / denom

    # Modules should be reordered according to decreasing "average degree".
    cluster_sizes = []
    average_degrees = []
    for selection in selections:
        cluster_sizes.append(selection.size)
        average_degrees.append(degrees[selection].mean())

    module_to_cluster = np.argsort(average_degrees)[::-1]
    cluster_to_module = {v : k for k, v in enumerate(module_to_cluster)}

    triples = [(
        cluster_to_module[clustering[i]],
        -degrees[i],
        i,
        ) for i in range(p)]

    _a, _b, new_to_old_idx = zip(*sorted(triples))

    # Make a csv file if requested.
    header = ('Gene', 'Module', 'Entry Index', 'Average Degree', 'Degree')
    with open(args.out, 'w') as fout:          # amm changed wb to w
        writer = csv.writer(fout, 'excel-tab') #problematic; need to switch to tsv file!
        writer.writerow(header)
        for old_i in new_to_old_idx:
            name = remaining_row_names[old_i]
            cluster = clustering[old_i]
            row = (
                    name,
                    cluster_to_module[cluster] + 1,
                    row_index_map[name] + 1,
                    average_degrees[cluster],
                    degrees[old_i],
                    )
            writer.writerow(row)

    #Create Output
    fh1=figureHandler(proj="2d")
    fh2=figureHandler(proj="2d")
    fh3=figureHandler(proj="2d")

    # Prepare to create the sorted heatmaps. (fh2)
    C_sorted = C[np.ix_(new_to_old_idx, new_to_old_idx)]
    clustering_new = clustering[np.ix_(new_to_old_idx)]

    # Draw the third heatmap (smoothed).
    # Make a smoothed correlation array. (fh3)
    S = expansion(clustering_new)
    block_mask = S.dot(S.T)
    denom = np.outer(S.sum(axis=0), S.sum(axis=0))
    small = S.T.dot(C_sorted).dot(S) / denom
    C_all_smoothed = S.dot(small).dot(S.T)
    C_smoothed = (C_all_smoothed * (1 - block_mask) + C_sorted * block_mask)

    # Getting list of names for heatmaps 2 and 3
    hpnames = [remaining_row_names[old_i] for old_i in new_to_old_idx]

    # Plot using something like http://stackoverflow.com/questions/15988413/
    # Drawing heatmaps
    # Draw first heatmap [C]
    hm.plotHeatmap(C,fh1.ax[0], cmap=palette.mpl_colormap, xlbls=remaining_row_names,
                    ylbls=remaining_row_names)
    fh1.formatAxis(xTitle="sampleID", figTitle="Correlations")

    # Draw second heatmap [C_sorted](reordered according to the clustering).
    hm.plotHeatmap(C_sorted,fh2.ax[0], cmap=palette.mpl_colormap, xlbls=hpnames,
                    ylbls=hpnames)
    fh2.formatAxis(xTitle="sampleID", figTitle="Re-Ordered correlations")

    # Draw the heatmap [C_smoothed](smoothed version of C_sorted)
    hm.plotHeatmap(C_smoothed,fh3.ax[0], cmap=palette.mpl_colormap, xlbls=hpnames,
                    ylbls=hpnames)
    fh3.formatAxis(xTitle="sampleID", figTitle="Smoothed correlations")

    #Create output from maps
    with PdfPages(args.figure) as pdf:
        fh1.addToPdf(pdf)
        fh2.addToPdf(pdf)
        fh3.addToPdf(pdf)


def main(args):
    # Import data through the SECIMTools interface
    dat = wideToDesign(wide=args.input,design=args.design,uniqID=args.uniqID, 
                        logger=logger)
    logger.info('Number of variables: {0}'.format(dat.wide.shape[0]))
    logger.info('Number of observations per variable: {0}'.format(dat.wide.shape[1]))

    ## If there is no variance in a row, the correlations cannot be computed.
    dat.wide["variance"]=dat.wide.apply(lambda x: ((x-x.mean()).sum()**2),axis=1)
    dat.wide            = dat.wide[dat.wide["variance"]!=0.0]
    dat.wide.drop("variance",axis=1,inplace=True)
    logger.info("Table arranged")

    # Compute the matrix of correlation coefficients.
    C = dat.wide.T.corr(method=args.correlation).values
    logger.info("Correlated")

    # For now, ignore the possibility that a variable
    # will have negligible variation.
    mask = np.ones(dat.wide.shape[0], dtype=bool)

    # Count the number of variables not excluded from the clustering.
    p = np.count_nonzero(mask)

    # Consider all values of tuning parameter sigma in this array.
    sigmas, step = np.linspace(args.sigmaLow, args.sigmaHigh, num=args.sigmaNum,
                                retstep=True)

    # Compute the clustering for each of the several values of sigma.
    # Each sigma corresponds to a different affinity matrix,
    # so the modularity matrix is also different for each sigma.
    # The goal is to the clustering whose modularity is greatest
    # across all joint (sigma, partition) pairs.
    # In practice, we will look for an approximation of this global optimum.
    #exit()
    logger.info("Begin clustering")
    clustering, sigma, m = get_clustering(C, sigmas)

    # Report a summary of the results of the technical analysis.
    logger.info("After partition refinement:")
    logger.info("Sigma: {0}".format(sigma))
    logger.info("Number of clusters: {0}".format(clustering.max()+1))
    logger.info("Modulated modularity: {0}".format(m))

    # Run the nontechnical analysis using the data frame and the less nerdy
    # of the outputs from the technical analysis.
    nontechnical_analysis(args, dat.wide, mask, C, clustering)
    logger.info("Script Complete!")


if __name__ == '__main__':
    args = getOptions()
    logger = logging.getLogger()
    sl.setLogger(logger)
    logger.info("Importing data with following parameters:"\
                "\n\tInput: {0}"\
                "\n\tDesign: {1}"\
                "\n\tuniqID: {2}".format(args.input, args.design, args.uniqID))
    palette = colorHandler(pal=args.palette, col=args.color)
    logger.info(u"Using {0} color scheme from {1} palette".format(args.color,
                args.palette))
    main(args)
