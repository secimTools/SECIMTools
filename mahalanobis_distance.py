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
from matplotlib.backends.backend_pdf import PdfPages

# Local packages
from interface import wideToDesign
import plotting


iteratorOfColors = 0
def getOptions():
    """ Function to pull in arguments """
    description = """ """
    parser = argparse.ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)

    group1 = parser.add_argument_group(description="Input Files")
    group1.add_argument("--input", dest="wide", action='store', required=True, help="Dataset in Wide format")
    group1.add_argument("--design", dest="design", action='store', required=True, help="Dataset in Design format")
    parser.add_argument("--ID", dest="uniqID", action='store', required=True, help="Name of the column with unique identifiers.")
    group1.add_argument("--group", dest="group", action='store', required=False, default='', help="Separate plots by group. Must specify valid group name")

    group2 = parser.add_argument_group(description="Output Files")
    group2.add_argument("-o", dest="outPDF", action='store', required=True, help="Output of Mahalanobis distance PDF of plots")
    group2.add_argument("--flags", dest="flags", action='store', required=True, help="Output flag file")

    args = parser.parse_args()
    return(args)


def mahalnobisDistance(wide):
    """ Calculate the Mahalanobis distance and return an array of distances.

            Arguments:
                :type wide: pandas.DataFrame
                :param wide: A wide formatted data frame with samples as columns and compounds as rows.

                :param type string: The name of the title of the plot if using group separation

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

    # # Calculate cutoffs
    # numParam = wide.shape[1]
    # c99 = np.sqrt(stats.chi2.ppf(0.995, numParam))
    # c97 = np.sqrt(stats.chi2.ppf(0.975, numParam))
    # c95 = np.sqrt(stats.chi2.ppf(0.95, numParam))
    # cutoffs = (('99.5%', c99), ('97.5%', c97), ('95.0%', c95))

    # TODO: Create a flag based on values greater than one of the cutoffs.
    # return plotMahalanobis(MD, cutoffs, title)
    return MD

def plotMahalanobis(MD, color):
    """ Plot the Mahalanobis distance plot.

                Arguments:
                    :type MD: numpy.array
                    :param MD: An array of distances.

                    :param tuple cutoffs: A tuple with label and cutoff. Used a tuple so that sort order would be maintained.

                    :param type string: The name of the title of the plot if using group separation

                Returns:
                    :return: Returns a figure object.
                    :rtype: matplotlib.pyplot.Figure.figure

                """



    # Plot MD as scatter plot
    ax.scatter(x=range(MD.size), y=MD, c=color)
    ax.set_xlabel('Mean')
    ax.set_ylabel('Difference')
    ax.legend()




def galaxySavefig(fig, fname):
    """ Take galaxy DAT file and save as fig """

    png_out = fname + '.png'
    fig.savefig(png_out)

    # shuffle it back and clean up
    data = file(png_out, 'rb').read()
    with open(fname, 'wb') as fp:
        fp.write(data)
    os.remove(png_out)


def main():
    """ Main Script """

    # Convert inputted wide file to dataframe
    # df_trans = pd.DataFrame.from_csv(args.wide, sep='\t').T
    dat = wideToDesign(args.wide, args.design, args.uniqID)
    df_trans = dat.wide.transpose()

    # Calculate cutoffs
    numParam = dat.wide.shape[1]
    c99 = np.sqrt(stats.chi2.ppf(0.995, numParam))
    c97 = np.sqrt(stats.chi2.ppf(0.975, numParam))
    c95 = np.sqrt(stats.chi2.ppf(0.95, numParam))
    cutoffs = (('99.5%', c99), ('97.5%', c97), ('95.0%', c95))

    #Import colors and markers
    colorsAndMarkers = plotting.ColorMarker()
    colorsAndMarkers = colorsAndMarkers.get_colors()

    listOfMD = []


    # Open a multiple page PDF for plots
    outPDF = PdfPages(args.outPDF)
    # If using groups:
    if not args.group is '':
        try:
            for title, group in dat.design.groupby(args.group):

                # Filter the transposed wide file into a new dataframe
                currentFrame = df_trans.reindex(index=group.index)

                # Save figure of the group
                MD = mahalnobisDistance(currentFrame)
                # galaxySavefig(fig=figure, fname=args.plot + '_' + title)
                listOfMD.append(MD)
        except KeyError:
            logger.error("{} is not a column name in the design file.".format(args.group))
        except Exception as e:
            logger.error("Error. {}".format(e))

    else:
        # If not using groups:
        figure = mahalnobisDistance(df_trans)
        outPDF.savefig(figure)



    global fig
    global ax
    global legendList
    legendList=[]
    # Create figure object with a single axis
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.suptitle('Mahalanobis Distance')
    # Add a horizontal line above 95% of the data
    colors = ['r', 'y', 'c', 'b', 'k', 'm', 'g']
    for i, val in enumerate(cutoffs):
        ax.axhline(val[1], color=colors[i], ls='--', lw=2, label='{} $\chi^2$'.format(val[0]))
    plt.legend()

    for color, md in zip(colors,listOfMD):
        plotMahalanobis(md,color)

    outPDF.savefig(fig)

    outPDF.close()
    # galaxySavefig(fig=figure, fname=args.plot)
    # figure.savefig(args.plot + '.png', bbox_inches='tight')



if __name__ == '__main__':
    # Turn on Logging if option -g was given

    global args
    args = getOptions()

    # Turn on logging
    logging.basicConfig()
    logger = logging.getLogger()
    # if args.debug:
    #     mclib.logger.setLogger(logger, args.log, 'debug')
    # else:
    #     mclib.logger.setLogger(logger, args.log)

    # Output git commit version to log, if user has access
    # mclib.git.git_to_log(__file__)

    # Run Main part of the script
    main()
    logger.info("Script complete.")
