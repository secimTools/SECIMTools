#!/usr/bin/env python

# Built-in packages
import argparse
from argparse import RawDescriptionHelpFormatter
import logging
import os

# Add-on packages
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from sklearn.neighbors import DistanceMetric

def getOptions():
    """ Function to pull in arguments """
    description = """ """
    parser = argparse.ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)

    group1 = parser.add_argument_group(description="Input Files")
    group1.add_argument("--input", dest="wide", action='store', required=True, help="Dataset in Wide format")

    group2 = parser.add_argument_group(description="Output Files")
    group2.add_argument("--o", dest="plot", action='store', required=True, help="Output of Mahalanobis distance plot")

    args = parser.parse_args()
    return(args)


def standardizedEulideanDistance(wide):
    """ Calculate the standardized Euclidean distance and return an array of distances to the center and a matrix of pairewise distances.

    Arguments:
        :type wide: pandas.DataFrame
        :param wide: A wide formatted data frame with samples as columns and compounds as rows.

    Returns:
        :return: Return a numpy array with MD values.
        :rtype: numpy.array
    """
    
    
    # Estimated Variance from the data
    varHat = wide.var(axis=1, ddof=1)
    dist = DistanceMetric.get_metric('seuclidean', V=varHat)

    # Column means
    colMean = wide.mean(axis=1)

    # Calculate the standardized Euclidean Distance from all samples to the center
    SEDtoCenter = dist.pairwise(wide.values.T, pd.DataFrame(colMean).T)
    
    # Calculate the pairwise standardized Euclidean Distance of all samples
    SEDpairwise = dist.pairwise(wide.values.T)

    # Calculate cutoffs
    pSamples  = float(wide.shape[1])
    nFeatures = float(wide.shape[0])
    nIterate  = 100000
    p = 0.95
    betaP     = np.percentile(pd.DataFrame(stats.beta.rvs(0.5, 0.5*(pSamples-2), size=nIterate*nFeatures).reshape(nIterate, nFeatures)).sum(axis=1), p*100)
    betaCut1  = np.sqrt((pSamples-1)**2/pSamples*betaP)
    normCut1  = np.sqrt(stats.norm.ppf(p, (pSamples-1)/pSamples*nFeatures, np.sqrt(2*nFeatures*(pSamples-2)*(pSamples-1)**2/pSamples**2/(pSamples+1))))
    chisqCut1 = np.sqrt(stats.chi2.ppf(p, nFeatures))
    betaCut2  = np.sqrt((pSamples-1)*2*betaP)
    normCut2  = np.sqrt(stats.norm.ppf(p, 2*nFeatures, np.sqrt(8*nFeatures*(pSamples-2)/(pSamples+1))))
    chisqCut2 = np.sqrt(2*stats.chi2.ppf(p, nFeatures))
    cutoff1   = (('Beta(Exact)', betaCut1), ('Normal', normCut1), ('Chi-square', chisqCut1))
    cutoff2   = (('Beta(Exact)', betaCut2), ('Normal', normCut2), ('Chi-square', chisqCut2))

    # TODO: Create a flag based on values greater than one of the cutoffs.
    return plotMahalanobis(SEDtoCenter, cutoff1, SEDpairwise, cutoff2)


def plotMahalanobis(SEDtoCenter, cutoff1, SEDpairwise, cutoff2):
    """ Plot the standardized Euclidean distance plot.

    Arguments:
        :type MD: numpy.array
        :param MD: An array of distances.

        :param tuple cutoffs: A tuple with label and cutoff. Used a tuple so that sort order would be maintained.

    Returns:
        :return: Returns a figure object.
        :rtype: matplotlib.pyplot.Figure.figure

    """
    # Create figure object with a single axis
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.suptitle('standardized Euclidean Distance')

    # Plot MD as scatter plot
    ax.scatter(x=range(len(SEDtoCenter)), y=SEDtoCenter)
    ax.set_xlabel('Index')
    ax.set_ylabel('standardized Euclidean Distance')

    # Add a horizontal line above 95% of the data
    colors = ['r', 'y', 'c', 'b', 'k', 'm', 'g']
    lslist = ['-', '--', '--']
    lwlist = [1, 2, 2]
    for i, val in enumerate(cutoff1):
        ax.axhline(val[1], color=colors[i], ls=lslist[i], lw=lwlist[i], label='{0} 95% Cutoff: {1}'.format(val[0],round(val[1],1)))
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    return fig


def galaxySavefig(fig, fname):
    """ Take galaxy DAT file and save as fig """

    png_out = fname + '.png'
    fig.savefig(png_out)

    # shuffle it back and clean up
    data = file(png_out, 'rb').read()
    with open(fname, 'wb') as fp:
        fp.write(data)
    os.remove(png_out)


def main(args):
    """ Main Script """

    # Convert inputted wide file to dataframe
    df_trans = pd.DataFrame.from_csv(args.wide, sep='\t')
    figure = standardizedEulideanDistance(df_trans)

    galaxySavefig(fig=figure, fname=args.plot)
    # figure.savefig(args.plot + '.png', bbox_inches='tight')


if __name__ == '__main__':
    # Turn on Logging if option -g was given
    args = getOptions()

    # Turn on logging
    logger = logging.getLogger()
    # if args.debug:
    #     mclib.logger.setLogger(logger, args.log, 'debug')
    # else:
    #     mclib.logger.setLogger(logger, args.log)

    # Output git commit version to log, if user has access
    # mclib.git.git_to_log(__file__)

    # Run Main part of the script
    main(args)
    logger.info("Script complete.")
