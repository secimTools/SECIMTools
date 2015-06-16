#!/usr/bin/env python

# Built-in packages
import argparse
from argparse import RawDescriptionHelpFormatter
import logging
import os

# Add-on packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats


def getOptions():
    """ Function to pull in arguments """
    description = """ """
    parser = argparse.ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)

    group1 = parser.add_argument_group(description="Input Files")
    group1.add_argument("--input", dest="wide", action='store', required=True, help="Dataset in Wide format")

    group2 = parser.add_argument_group(description="Output Files")
    group2.add_argument("-o", dest="plot", action='store', required=True, help="Output of Mahalanobis distance plot")

    args = parser.parse_args()
    return(args)


def mahalnobisDistance(wide):
    """ Calculate the Mahalanobis distance and return an array of distances.

    Arguments:
        :type wide: pandas.DataFrame
        :param wide: A wide formatted data frame with samples as columns and compounds as rows.

    Returns:
        :return: Return a numpy array with MD values.
        :rtype: numpy.array
    """
    # Covariance matrix from the data
    covHat = wide.cov()

    # Inverse of the covariance matrix
    covHatInv = np.linalg.inv(covHat)

    # Column means
    colMean = wide.mean(axis=0)

    # Mean Center the data
    center = (wide - colMean).T

    # Calculate the mahalanobis distance sqrt{(Yi - Ybar)^T Sigma^-1 (Yi - Ybar)}
    MD = np.diag(np.sqrt(np.dot(np.dot(center.T, covHatInv), center)))

    # TODO: Maybe output the MD file in the future?
    return plotMahalanobis(MD, numParam=wide.shape[1])


def plotMahalanobis(MD, numParam):
    """ Plot the Mahalanobis distance plot.

    Arguments:
        :type MD: numpy.array
        :param MD: An array of distances.

    Returns:
        :return: Returns a figure object.
        :rtype: matplotlib.pyplot.Figure.figure

    """
    # Create figure object with a single axis
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.suptitle('Mahalanobis Distance')

    # Plot MD as scatter plot
    ax.scatter(x=range(len(MD)), y=MD)
    ax.set_xlabel('Mean')
    ax.set_ylabel('Difference')

    # Add a horizontal line above 95% of the data
    ax.axhline(np.sqrt(stats.chi2.ppf(0.995, numParam)), color='r', ls='--', lw=2, label='99.5% $\chi^2$'.format(numParam))
    ax.axhline(np.sqrt(stats.chi2.ppf(0.975, numParam)), color='y', ls='--', lw=2, label='97.5% $\chi^2$'.format(numParam))
    ax.axhline(np.sqrt(stats.chi2.ppf(0.95, numParam)), color='c', ls='--', lw=2, label='95% $\chi^2$'.format(numParam))
    plt.legend()
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
    df_wide = pd.DataFrame.from_csv(args.wide, sep='\t')
    figure = mahalnobisDistance(df_wide)

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
