#!/usr/bin/env python

# Built-in packages
import argparse
from argparse import RawDescriptionHelpFormatter
import logging

# Add-on packages
import numpy as np
import matplotlib.pyplot as plt

# McLab packages
import mclib


def getOptions():
    """ Function to pull in arguments """
    #TODO: Add cmd line options, I just copied and pasted these.
    description = """ """
    parser = argparse.ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)

    group1 = parser.add_argument_group(description="Input Files")
    group1.add_argument("--vcf", dest="vcfName", action='store', required=True, help="Name of VCF file. If not zipped using bgzip will try to create zip. [Required]")

    group2 = parser.add_argument_group(description="Output Files")
    group2.add_argument("-o", dest="oname", action='store', required=True, help="Name of output FASTA. [Required]")
    group2.add_argument("--log", dest="log", action='store', required=False, help="Name of the LOG file [Optional]")
    parser.add_argument("--debug", dest="debug", action='store_true', required=False, help="Enable debug output.")

    args = parser.parse_args()
    return(args)


def mahalnobisDistance(wide):
    """ Calculate the Mahalanobis istance and return an array of distances.

    Arguments:
        :type wide: pandas.DataFrame
        :param wide: A wide formated data frame with samples as columns and compounds as rows.

    Returns:
        :return: Return a numpy array with MD values.
        :rtype: numpy.array
    """
    # Covaranice matrix from the data
    covHat = wide.cov()

    # Inverse of the covarance matrix
    covHatInv = np.linalg.inv(covHat)

    # Column means
    colMean = wide.mean(axis=0)

    # Mean Center the data
    center = (wide - colMean).T

    # Calculate the mahalanobis distance sqrt{(Yi - Ybar)^T Sigma^-1 (Yi - Ybar)}
    MD = np.diag(np.sqrt(np.dot(np.dot(center.T, covHatInv), center)))

    return MD


def plotMahalanobis(MD):
    """ Plot the Mahalanobis distance plot.

    Arguments:
        :type MD: numpy.array
        :param MD: An array of distances.

    Returns:
        :return: Returns a figure object.
        :rtype: matplotlib.figure
    """
    # Create figure object with a single axis
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot MD as scatter plot
    ax.scatter(x=range(len(MD)), y=MD)

    # Add a horizontal line above 95% of the data
    ax.axhline(np.percentile(MD, 95), ls='--', lw=2)

    #TODO: Instead of returning the figure we could go ahead and write it out to a file.

    return fig


def main(args):
    """ Main Script """
    #TODO: Add the main arguments

if __name__ == '__main__':
    # Turn on Logging if option -g was given
    args = getOptions()

    # Turn on logging
    logger = logging.getLogger()
    if args.debug:
        mclib.logger.setLogger(logger, args.log, 'debug')
    else:
        mclib.logger.setLogger(logger, args.log)

    # Output git commit version to log, if user has access
    mclib.git.git_to_log(__file__)

    # Run Main part of the script
    main(args)
    logger.info("Script complete.")
