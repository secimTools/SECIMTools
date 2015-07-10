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
from matplotlib.backends.backend_pdf import PdfPages

# Local packages
from interface import wideToDesign
from interface import Flags

def getOptions():
    """ Function to pull in arguments """
    description = """ """
    parser = argparse.ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)

    group1 = parser.add_argument_group(description="Required Input")
    group1.add_argument("--input", dest="fname", action='store', required=True, help="Dataset in Wide format")
    group1.add_argument("--design", dest="dname", action='store', required=True, help="Design file")
    group1.add_argument("--ID", dest = "uniqID", action = 'store', required = True, help = "Name of the column with unique identifiers.")
    group1.add_argument("--group", dest="group", action='store', required=False, help="Treatment group")
    group1.add_argument("--SEDplotOutFile", dest="plot", action='store', required=True, help="PDF Output of standardized Euclidean distance plot")
    group1.add_argument("--SEDtoCenter", dest="out1", action='store', required=True, help="TSV Output of standardized Euclidean distances from samples to the center")
    group1.add_argument("--SEDpairwise", dest="out2", action='store', required=True, help="TSV Output of sample-pairwise standardized Euclidean distances")

    group2 = parser.add_argument_group(description="Optional Input")
    group2.add_argument("--p", dest="p", action='store', required=False, default=0.95, type=float, help="The percentile cutoff for standard distributions. The default is 0.95. ")

    args = parser.parse_args()
    return(args)

def saveFigToPdf(pdf, SEDtoCenter, cutoff1, SEDpairwise, cutoff2, groupName, p):
    
    fig1 = plotSEDtoCenter(SEDtoCenter, cutoff1, groupName, p)
    pdf.savefig(fig1, bbox_inches='tight')
    plt.close(fig1)
    
    fig2 = scatterPlotSEDpairwise(SEDpairwise, cutoff2, groupName, p)
    pdf.savefig(fig2, bbox_inches='tight')
    plt.close(fig2)
    
    fig3 = boxPlotSEDpairwise(SEDpairwise, cutoff2, groupName, p)
    pdf.savefig(fig3, bbox_inches='tight')
    plt.close(fig3)
    
    return pdf

def SEDbyGroup(dat, wide, args):

    pdfOut = PdfPages(args.plot)
    
    if not args.group:
        SEDtoCenter, cutoffForSEDtoCenter, SEDpairwise, cutoffForSEDpairwise = standardizedEulideanDistance(wide, args.p)
        saveFigToPdf(pdfOut, SEDtoCenter, cutoffForSEDtoCenter, SEDpairwise, cutoffForSEDpairwise, '', args.p)
    else:
        SEDtoCenter = pd.DataFrame(columns=['SED_to_Center', 'group'])
        SEDpairwise = pd.DataFrame(columns=['group'])
        cutoffForSEDtoCenter = pd.DataFrame(columns = ['Beta(Exact)', 'Normal', 'Chi-sq'])
        cutoffForSEDpairwise = pd.DataFrame(columns = ['Beta(Exact)', 'Normal', 'Chi-sq'])
        for title, group in dat.design.groupby(args.group):
            # Filter the wide file into a new dataframe
            currentFrame = wide[group.index]

            # Change dat.sampleIDs to match the design file
            dat.sampleIDs = group.index
            
            SEDtoCenter_group, cutoff1, SEDpairwise_group, cutoff2 = standardizedEulideanDistance(currentFrame, args.p)
            pdfOut = saveFigToPdf(pdfOut, SEDtoCenter_group, cutoff1, SEDpairwise_group, cutoff2, 'in group ' + title, args.p)
            
            SEDtoCenter_group['group'] = [title]*currentFrame.shape[1]
            SEDpairwise_group['group'] = [title]*currentFrame.shape[1]
            SEDtoCenter = pd.DataFrame.merge(SEDtoCenter, SEDtoCenter_group, on=['SED_to_Center', 'group'], left_index=True, right_index=True, how='outer', sort=False)
            SEDpairwise = pd.DataFrame.merge(SEDpairwise, SEDpairwise_group, on='group', left_index=True, right_index=True, how='outer', sort=False)
            cutoffForSEDtoCenter.loc[title] = cutoff1.values[0]
            cutoffForSEDpairwise.loc[title] = cutoff2.values[0]
            
        cutoffForSEDtoCenter.loc['mean'] = cutoffForSEDtoCenter.mean()
        cutoffForSEDpairwise.loc['mean'] = cutoffForSEDpairwise.mean()
        #SEDtoCenter = SEDtoCenter.sort('group')
        #SEDpairwise = SEDpairwise.sort('group')
        #SEDpairwise = SEDpairwise.reindex
        saveFigToPdf(pdfOut, SEDtoCenter.drop('group',axis=1), cutoffForSEDtoCenter.loc[['mean']], SEDpairwise.drop('group',axis=1), cutoffForSEDpairwise.loc[['mean']], '', args.p)
    
    pdfOut.close()
    
    SEDtoCenter.to_csv(args.out1, sep='\t')
    SEDpairwise.to_csv(args.out2, sep='\t')
    
    return SEDtoCenter, SEDpairwise

def standardizedEulideanDistance(wide, p):
    """ Calculate the standardized Euclidean distance and return an array of distances to the center and a matrix of pairwise distances.

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
    SEDtoCenter = pd.DataFrame(SEDtoCenter, columns = ['SED_to_Center'], index = wide.columns)
    
    # Calculate the pairwise standardized Euclidean Distance of all samples
    SEDpairwise = dist.pairwise(wide.values.T)
    SEDpairwise = pd.DataFrame(SEDpairwise, columns = wide.columns, index = wide.columns)
    for index, row in SEDpairwise.iterrows():
        SEDpairwise.loc[index, index] = np.nan

    # Calculate cutoffs
    pSamples  = float(wide.shape[1])
    nFeatures = float(wide.shape[0])
    nIterate  = 20000 #100000
    #p = 0.95
    betaP     = np.percentile(pd.DataFrame(stats.beta.rvs(0.5, 0.5*(pSamples-2), size=nIterate*nFeatures).reshape(nIterate, nFeatures)).sum(axis=1), p*100)
    betaCut1  = np.sqrt((pSamples-1)**2/pSamples*betaP)
    normCut1  = np.sqrt(stats.norm.ppf(p, (pSamples-1)/pSamples*nFeatures, np.sqrt(2*nFeatures*(pSamples-2)*(pSamples-1)**2/pSamples**2/(pSamples+1))))
    chisqCut1 = np.sqrt((pSamples-1)/pSamples*stats.chi2.ppf(p, nFeatures))
    betaCut2  = np.sqrt((pSamples-1)*2*betaP)
    normCut2  = np.sqrt(stats.norm.ppf(p, 2*nFeatures, np.sqrt(8*nFeatures*(pSamples-2)/(pSamples+1))))
    chisqCut2 = np.sqrt(2*stats.chi2.ppf(p, nFeatures))
    cutoff1   = pd.DataFrame([[betaCut1, normCut1, chisqCut1]], columns=['Beta(Exact)', 'Normal', 'Chi-sq'])
    cutoff2   = pd.DataFrame([[betaCut2, normCut2, chisqCut2]], columns=['Beta(Exact)', 'Normal', 'Chi-sq'])

    # TODO: Create a flag based on values greater than one of the cutoffs.
    #return plotSED(SEDtoCenter, cutoff1, SEDpairwise, cutoff2)
    return SEDtoCenter, cutoff1, SEDpairwise, cutoff2

def addCutoff(fig, ax, cutoff, p):

    colors = ['r', 'y', 'c', 'b', 'k', 'm', 'g']
    lslist = ['-', '--', '--']
    lwlist = [1, 2, 2]
    val = cutoff.values[0]
    col = cutoff.columns
    for i in [0, 1, 2]:
        ax.axhline(val[i], color=colors[i], ls=lslist[i], lw=lwlist[i], label='{0} {1}% Cutoff: {2}'.format(col[i],round(p*100,3),round(val[i],1)))
    
    return fig, ax

def figInitiate(figWidth, colNames, figTitle):

    fig, ax = plt.subplots(1, 1, figsize=(figWidth, 8))
    fig.suptitle(figTitle)
    ax.set_xlim(-0.5, -0.5+len(colNames))
    plt.xticks(range(len(colNames)), colNames, rotation=min(90,max(0,((len(colNames)-1)/4-1)*30)))
    
    return fig, ax

def plotSEDtoCenter(SEDtoCenter, cutoff, groupName, p):
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
    fig, ax = figInitiate(max(SEDtoCenter.shape[0]/4, 12), SEDtoCenter.index, 'standardized Euclidean Distance from samples {} to the center'.format(groupName))

    # Plot SED from samples to the center as scatter plot
    ax.scatter(x=range(len(SEDtoCenter.values)), y=SEDtoCenter.values)
    ax.set_xlabel('Index')
    ax.set_ylabel('standardized Euclidean Distance')

    # Add a horizontal line above 95% of the data
    fig, ax = addCutoff(fig, ax, cutoff, p)
    plt.legend(fancybox=True, framealpha=0.5)
    
    return fig

def scatterPlotSEDpairwise(SEDpairwise, cutoff, groupName, p):

    # Plot pairwise SED for samples
    #SEDpairwise.index = range(SEDpairwise.shape[0])
    fig, ax = figInitiate(max(SEDpairwise.shape[0]/4, 12), SEDpairwise.columns, 'pairwise standardized Euclidean Distance from samples {}'.format(groupName))
    
    for i in range(SEDpairwise.shape[0]):
        ax.scatter(x=[i]*SEDpairwise.shape[1], y=SEDpairwise.ix[i].values, marker='+')
        ax.set_ylabel('standardized Euclidean Distance')
        
    # Add a horizontal line above 95% of the data
    fig, ax = addCutoff(fig, ax, cutoff, p)
    plt.legend(fancybox=True, framealpha=0.5)
    
    return fig
    
def boxPlotSEDpairwise(SEDpairwise, cutoff, groupName, p):
    
    fig, ax = figInitiate(max(SEDpairwise.shape[0]/4, 12), SEDpairwise.columns, 'Box-plots for pairwise standardized Euclidean Distance from samples {}'.format(groupName))
    
    ax.boxplot(SEDpairwise.values, showmeans=True, labels=SEDpairwise.columns)
    
    # Add a horizontal line above 95% of the data
    fig, ax = addCutoff(fig, ax, cutoff, p)
    plt.legend(fancybox=True, framealpha=0.5)
    
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

    # Import data
    dat = wideToDesign(args.fname, args.dname, args.uniqID)
    
    # Only interested in samples
    wide = dat.wide[dat.sampleIDs]
    
    # Calculate SED by group or not
    SEDbyGroup(dat, wide, args)

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
