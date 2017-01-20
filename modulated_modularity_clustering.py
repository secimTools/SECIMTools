#!/usr/bin/env python
################################################################################
# DATE: 2017/01/17
# 
# MODULE: modulated_modularity_clustering.py
#
# VERSION: 1.1
# 
# EDITED: Miguel Ibarra (miguelib@ufl.edu) Matt Thoburn (mthoburn@ufl.edu) 
#
# DESCRIPTION: This tool runs modulated modularity clustering (mmc)
#
################################################################################
# Built-in packages
import csv
import logging
import argparse
from contextlib import contextmanager

# Add on modules
import scipy
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from matplotlib.backends.backend_pdf import PdfPages

# Local Packages
import logger as sl
from interface import wideToDesign

# Graphing packages
import module_heatmap as hm
from manager_figure import figureHandler
from module_mmc import expansion, get_clustering


def getOptions(myOpts = None):
    # Get command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",'--correlation', 
                        choices=('pearson', 'kendall', 'spearman'),
                        default='pearson',
                        help=("Compute correlation coefficients using either "
                        "'pearson' (standard correlation coefficient), "
                        "'kendall' (Kendall Tau correlation coefficient), or "
                        "'spearman' (Spearman rank correlation)."))
    parser.add_argument("-i",'--input', required=True,
                        help=('Path to the data file, which is expected to be in'
                        'tabular (tsv) format with row and column labels, '
                        'and for which the rows are to be clustered.'))
    parser.add_argument("-d","--design", dest="design", required=True, 
                        help="Path of design file")
    parser.add_argument("-id","--ID",dest="uniqID",required=True, 
                        help="Unique Identifier")
    parser.add_argument("-sl",'--sigmaLow', type=float, default=0.05,
                        help='Low value of sigma (Default: 0.05).')
    parser.add_argument("-sh",'--sigmaHigh', type=float, default=0.50,
                        help='High value of sigma (Default: 0.50).')
    parser.add_argument("-sn",'--sigmaNum', type=float, default=451,
                        help='Number of values of sigma to search (Default: 451).')

    # Specify output file names relative to the directory
    # containing the input csv file.
    parser.add_argument('-f', "--figure", dest="figure", required=True,
                        help="MMC Heatmaps")
    parser.add_argument('-o',"--out", dest="out", required=True, 
                        help="Output TSV name")

    args = parser.parse_args()
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
    with open(args.out, 'wb') as fout:
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
    hm.plotHeatmap(C,fh1.ax[0],xlbls=remaining_row_names,ylbls=remaining_row_names)
    fh1.formatAxis()

    # Draw second heatmap [C_sorted](reordered according to the clustering).
    hm.plotHeatmap(C_sorted,fh2.ax[0],xlbls=hpnames,ylbls=hpnames)
    fh2.formatAxis()

    # Draw the heatmap [C_smoothed](smoothed version of C_sorted)
    hm.plotHeatmap(C_smoothed,fh3.ax[0],xlbls=hpnames,ylbls=hpnames)
    fh3.formatAxis()

    #Create output from maps
    with PdfPages(args.figure) as pdf:
        fh1.addToPdf(pdf)
        fh2.addToPdf(pdf)
        fh3.addToPdf(pdf)

def main(args):
    # Import data through the SECIMTools interface 
    data = wideToDesign(wide=args.input,design=args.design,uniqID=args.uniqID)
    logger.info('Number of variables: {0}'.format(data.wide.shape[0]))
    logger.info('Number of observations per variable: {0}'.format(data.wide.shape[1]))

    ## If there is no variance in a row, the correlations cannot be computed.
    data.wide["variance"]=data.wide.apply(lambda x: ((x-x.mean()).sum()**2),axis=1)
    data.wide = data.wide[data.wide["variance"]!=0.0]
    data.wide.drop("variance",axis=1,inplace=True)
    logger.info("Table arranged")

    # Compute the matrix of correlation coefficients.
    C = data.wide.T.corr(method=args.correlation).values
    logger.info("Correlated")
    
    # For now, ignore the possibility that a variable
    # will have negligible variation.
    mask = np.ones(data.wide.shape[0], dtype=bool)

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
    nontechnical_analysis(args, data.wide, mask, C, clustering)

if __name__ == '__main__':
    args = getOptions()

    # Activate Logger
    logger = logging.getLogger()
    sl.setLogger(logger)

    # Starting program
    main(args)
