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
    group1.add_argument("--SEDplotOutFile", dest="plot", action='store', required=True, help="PDF Output of standardized Euclidean distance plot")
    group1.add_argument("--SEDtoCenter", dest="out1", action='store', required=True, help="TSV Output of standardized Euclidean distances from samples to the center")
    group1.add_argument("--SEDpairwise", dest="out2", action='store', required=True, help="TSV Output of sample-pairwise standardized Euclidean distances")

    group2 = parser.add_argument_group(description="Optional Input")
    group1.add_argument("--group", dest="group", action='store', required=False, help="Treatment group")
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

def sortByCols(DF, colNames):
    
    sortedDF = pd.DataFrame()
    for colName in colNames:
        sortedDF[colName] = DF[colName]
    return sortedDF

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
        SEDtoCenter = SEDtoCenter.sort('group')
        SEDpairwise = SEDpairwise.sort('group')
        SEDpairwise = sortByCols(SEDpairwise, [SEDpairwise.index,'group'])
        saveFigToPdf(pdfOut, SEDtoCenter, cutoffForSEDtoCenter.loc[['mean']], SEDpairwise, cutoffForSEDpairwise.loc[['mean']], '', args.p)
    
    pdfOut.close()
    
    SEDtoCenter.drop('group',axis=1).to_csv(args.out1, sep='\t')
    SEDpairwise.drop('group',axis=1).to_csv(args.out2, sep='\t')
    
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
    return SEDtoCenter, cutoff1, SEDpairwise, cutoff2

def addCutoff(fig, ax, cutoff, p):

    colors = ['r', 'y', 'c']
    lslist = ['-', '--', '--']
    lwlist = [1, 2, 2]
    val = cutoff.values[0]
    col = cutoff.columns
    for i in [0, 1, 2]:
        ax.axhline(val[i], color=colors[i], ls=lslist[i], lw=lwlist[i], label='{0} {1}% Cutoff: {2}'.format(col[i],round(p*100,3),round(val[i],1)))
    
    return fig, ax

def getRot(n):
    
    return min(90,max(0,((n-1)/4-1)*30))

def figInitiate(figWidth, colNames, figTitle):

    fig, ax = plt.subplots(1, 1, figsize=(figWidth, 8))
    fig.suptitle(figTitle)
    ax.set_xlim(-0.5, -0.5+len(colNames))
    plt.xticks(range(len(colNames)), colNames)
    
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
    pSamples = SEDtoCenter.shape[0]
    fig, ax = figInitiate(max(pSamples/4, 12), SEDtoCenter.index, 'standardized Euclidean Distance from samples {} to the center'.format(groupName))

    # Add a horizontal line above 95% of the data
    fig, ax = addCutoff(fig, ax, cutoff, p)
    
    # Plot SED from samples to the center as scatter plot
    SEDtoCenter['ind'] = range(pSamples)
    if 'group' in SEDtoCenter.columns:
        kGroups = len(pd.Categorical(SEDtoCenter['group']).categories)
        groups = SEDtoCenter.groupby('group')
        colors = pd.tools.plotting._get_standard_colors(kGroups)
        colorindex = 0
        for name, group in groups:
            group.plot(kind='scatter', x='ind', y='SED_to_Center', marker='o', label=name, ax=ax, color=colors[colorindex], rot=getRot(pSamples))
            colorindex += 1
    else:
        SEDtoCenter.plot(kind='scatter', x='ind', y='SED_to_Center', marker='o', ax=ax, rot=getRot(pSamples))
    SEDtoCenter.drop('ind', axis=1, inplace=True)
    ax.set_xlabel('Index')
    ax.set_ylabel('standardized Euclidean Distance')
    
    plt.legend(fancybox=True, framealpha=0.5)
    plt.close(fig)
    
    return fig

def scatterPlotSEDpairwise(SEDpairwise, cutoff, groupName, p):

    # Plot pairwise SED for samples
    pSamples = SEDpairwise.shape[0]
    fig, ax = figInitiate(max(pSamples/4, 12), SEDpairwise.columns, 'pairwise standardized Euclidean Distance from samples {}'.format(groupName))
    
    # Add a horizontal line above 95% of the data
    fig, ax = addCutoff(fig, ax, cutoff, p)
    
    if 'group' in SEDpairwise.columns:
        colorindex = pd.Categorical(SEDpairwise['group']).categories.get_indexer(SEDpairwise['group'])
        group = SEDpairwise['group']
        kGroups = len(pd.Categorical(SEDpairwise['group']).categories)
    else:
        colorindex = [0]*pSamples
        group = ['pairwise SED']*pSamples
        kGroups = 1
    colors = pd.tools.plotting._get_standard_colors(kGroups)
    groupLabelled = []
    for i in range(pSamples):
        XY = pd.DataFrame([[i]*pSamples], index=['ind'], columns=SEDpairwise.index).T
        XY['SED'] = SEDpairwise.ix[:,i].values
        if (group[i] not in groupLabelled) and ('group' in SEDpairwise.columns):
            XY.plot(kind='scatter', x='ind', y='SED', marker='+', color=colors[colorindex[i]], label=group[i], ax=ax, rot=getRot(pSamples))
        else:
            XY.plot(kind='scatter', x='ind', y='SED', marker='+', color=colors[colorindex[i]], ax=ax, rot=getRot(pSamples))
        groupLabelled.append(group[i])
        ax.set_ylabel('standardized Euclidean Distance')
        
    plt.legend(fancybox=True, framealpha=0.5)
    plt.close(fig)
    
    return fig
    
def boxPlotSEDpairwise(SEDpairwise, cutoff, groupName, p):
    
    pSamples = SEDpairwise.shape[0]
    
    if 'group' in SEDpairwise.columns:
        #fig, ax = figInitiate(max(pSamples/4, 12), SEDpairwise.drop('group',axis=1).columns, 'Box-plots for pairwise standardized Euclidean Distance from samples {}'.format(groupName))
        fig, ax = plt.subplots(1, 1, figsize=(max(pSamples/4, 12), 8))
        fig.suptitle('Box-plots for pairwise standardized Euclidean Distance from samples {}'.format(groupName))
        kGroups = len(pd.Categorical(SEDpairwise['group']).categories)
        groups = SEDpairwise.groupby('group')
        colors = pd.tools.plotting._get_standard_colors(kGroups)
        posi = 0
        colorindex = 0
        for name, group in groups:
            boxplots = group.drop('group',axis=1).T.boxplot(ax=ax, rot=getRot(pSamples), return_type='dict', showmeans=True, positions=range(posi, posi+group.shape[0]), manage_xticks=False, patch_artist=True)
            for whisker in boxplots['whiskers']:
                whisker.set(color=colors[colorindex])
            for box in boxplots['boxes']:
                box.set(facecolor='w')
            for cap in boxplots['caps']:
                cap.set(color=colors[colorindex])
            for flier in boxplots['fliers']:
                flier.set(color=colors[colorindex], alpha=0.5)
            posi += group.shape[0]
            colorindex += 1
        plt.xticks(range(pSamples), group.drop('group',axis=1).columns)
        ax.set_xlim(-0.5, -0.5+pSamples)
    else:
        fig, ax = figInitiate(max(pSamples/4, 12), SEDpairwise.columns, 'Box-plots for pairwise standardized Euclidean Distance from samples {}'.format(groupName))
        SEDpairwise.boxplot(ax=ax, rot=getRot(pSamples), return_type='dict', showmeans=True, positions=range(0, pSamples), manage_xticks=False)
    
    # Add a horizontal line above 95% of the data
    fig, ax = addCutoff(fig, ax, cutoff, p)
    
    plt.legend(fancybox=True, framealpha=0.5)
    plt.close(fig)
    
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
