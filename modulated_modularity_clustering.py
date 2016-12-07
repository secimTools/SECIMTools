
from contextlib import contextmanager
import argparse
import csv
#from functools import partial
#import shutil as sh

import pandas as pd
import numpy as np

import scipy
from numpy.testing import assert_allclose

from module_mmc import expansion, get_clustering
import logging
import logger as sl
import module_heatmap as hm
from interface import wideToDesign
from manager_figure import figureHandler
from matplotlib.backends.backend_pdf import PdfPages


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
                        'tabular (tsv) format '
                        'with row and column labels, and for which the rows '
                        'are to be clustered.'))
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
    # Draw the first heatmap.
    # Plot using something like http://stackoverflow.com/questions/15988413/
    hm.plotHeatmap(C,fh1.ax[0])

    # Prepare to create the sorted heatmaps.
    C_new = C[np.ix_(new_to_old_idx, new_to_old_idx)]
    clustering_new = clustering[np.ix_(new_to_old_idx)]

    # Draw the second heatmap (reordered according to the clustering).
    hm.plotHeatmap(C_new,fh2.ax[0])

    # Draw the third heatmap (smoothed).
    # Make a smoothed correlation array.
    S = expansion(clustering_new)
    block_mask = S.dot(S.T)
    denom = np.outer(S.sum(axis=0), S.sum(axis=0))
    small = S.T.dot(C_new).dot(S) / denom
    C_all_smoothed = S.dot(small).dot(S.T)
    C_smoothed = (
            C_all_smoothed * (1 - block_mask) +
            C_new * block_mask)

    # Draw the heatmap.
    hm.plotHeatmap(C_smoothed,fh3.ax[0])
    #Create output from maps
    with PdfPages(args.figure) as pdf:
        fh1.addToPdf(pdf)
        fh2.addToPdf(pdf)
        fh3.addToPdf(pdf)

def main(args):

    data = wideToDesign(args.input,args.design,args.uniqID)
    df = data.wide
   
    p, n = df.shape
    logger.info('original number of variables in the input file: ' + str(p))
    logger.info('number of observations per variable: ' + str(n))

    ## If there is no variance in a row, the correlations cannot be computed.
    drops = []
    i = 0
    for i in range(0,df.shape[0]):
        if ((df.iloc[i]-df.iloc[i].mean()).sum()**2)==0.0:
            drops.append(i)
    df = df.drop(df.index[drops])
 
    logger.info("Table arranged\n")
    # Compute the matrix of correlation coefficients.
    C = df.T.corr(method=args.correlation).values
    
    logger.info("Correlated\n")
    
    # For now, ignore the possibility that a variable
    # will have negligible variation.
    mask = np.ones(df.shape[0], dtype=bool)

    # Count the number of variables not excluded from the clustering.
    p = np.count_nonzero(mask)

    # Consider all values of tuning parameter sigma in this array.
    sigmas, step = np.linspace(
            args.sigmaLow, args.sigmaHigh, num=args.sigmaNum, retstep=True)

    # Compute the clustering for each of the several values of sigma.
    # Each sigma corresponds to a different affinity matrix,
    # so the modularity matrix is also different for each sigma.
    # The goal is to the clustering whose modularity is greatest
    # across all joint (sigma, partition) pairs.
    # In practice, we will look for an approximation of this global optimum.
    
    logger.info("Begin clustering\n")

    clustering, sigma, m = get_clustering(C, sigmas)

    # Count the number of clusters.
    k = clustering.max() + 1

    # Report a summary of the results of the technical analysis.
    logger.info('after partition refinement:')
    logger.info('  sigma: ' + str(sigma))
    logger.info('  number of clusters: ' + str(k))
    logger.info('  modulated modularity: ' + str(m))


    # Run the nontechnical analysis using the data frame and the less nerdy
    # of the outputs from the technical analysis.
    nontechnical_analysis(args, df, mask, C, clustering)

if __name__ == '__main__':
    args = getOptions()

    # Activate Logger
    logger = logging.getLogger()

    sl.setLogger(logger)
    main(args)
