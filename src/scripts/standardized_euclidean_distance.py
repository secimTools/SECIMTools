#!/usr/bin/env python
##################################################################
# AUTHOR: Miguel A. Ibarra-Arellano (miguelib@ufl.edu)
#
# DESCRIPTION: Pairwise and mean standarized euclidean comparisons
##################################################################

import os
import logging
import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.neighbors import DistanceMetric
from secimtools.dataManager import logger as sl
from secimtools.dataManager.interface import wideToDesign
from secimtools.visualManager import module_box as box
from secimtools.visualManager import manager_color as ch
from secimtools.visualManager import module_lines as lines
from secimtools.visualManager import module_scatter as scatter
from secimtools.visualManager.manager_color import colorHandler
from secimtools.visualManager.manager_figure import figureHandler


def getOptions():
    """ Function to pull in arguments """
    description = """For mutivariate data, the standardized Euclidean distance
    (SED) takes the variance into consideration. If we assume all the vectors
    are identically and independently multinormally distributed with a diagonal
    covariance matrix (assuming the correlations are 0), the SED follows a
    certain distribution associated with the dimension of the vector. Thus,
    samples with SEDs higher than the cutoff are dubious and conflict with the
    multinormal assumption.

    The scripts estimate the variance according to the input data and calculate
    the SEDs from all samples to the estimated Mean and also the sample pairwise
    SEDs by group.

    The output includes 3 plots for each group and 3 plots in the end for all
    the samples altogether.
    """

    parser = argparse.ArgumentParser(description=description, formatter_class=
                                    argparse.RawDescriptionHelpFormatter)
    # Standard Input
    standard = parser.add_argument_group(description="Standard Input")
    standard.add_argument("-i","--input", dest="input", action='store',required=True,
                         help="Dataset in Wide format")
    standard.add_argument("-d","--design", dest="design", action='store',
                         required=True, help="Design file")
    standard.add_argument("-id","--ID", dest="uniqID", action='store', required=True,
                         help="Name of the column with unique identifiers.")
    standard.add_argument("-g","--group", dest="group",default=False, action='store',
                        required=False, help="Treatment group")
    standard.add_argument("-o","--order",dest="order",action="store",
                        default=False, help="Run Order")
    standard.add_argument("-l","--levels",dest="levels",action="store",
                        required=False, default=False, help="Different groups to"\
                        " sort by separeted by comas.")
    # Tool Output
    output = parser.add_argument_group(description="Output Files")
    output.add_argument("-f", "--figure", dest="figure", action='store',
                        required=True, help="PDF Output of standardized "\
                        "Euclidean distance plot")
    output.add_argument("-m","--SEDtoMean", dest="toMean", action='store',
                        required=True, help="TSV Output of standardized "
                        "Euclidean distances from samples to the mean.")
    output.add_argument("-pw","--SEDpairwise", dest="pairwise", action='store',
                        required=True, help="TSV Output of sample-pairwise "
                        "standardized Euclidean distances.")
    # Tool Input
    tool = parser.add_argument_group(description="Tool Input")
    tool.add_argument("-p","--per", dest="p", action='store', required=False,
                        default=0.95, type=float, help="The threshold "
                        "for standard distributions. The default is 0.95.")
    # Plot Options
    plot = parser.add_argument_group(title='Plot options')
    plot.add_argument("-pal","--palette",dest="palette",action='store',required=False,
                        default="tableau", help="Name of the palette to use.")
    plot.add_argument("-col","--color",dest="color",action="store",required=False,
                        default="Tableau_20", help="Name of a valid color scheme"\
                        " on the selected palette")
    args = parser.parse_args()

    # Standardize paths
    args.input    = os.path.abspath(args.input)
    args.design   = os.path.abspath(args.design)
    args.figure   = os.path.abspath(args.figure)
    args.toMean   = os.path.abspath(args.toMean)
    args.pairwise = os.path.abspath(args.pairwise)

    # Split levels if levels
    if args.levels:
        args.levels = args.levels.split(",")

    return(args)


def plotCutoffs(cut_S,ax,p):
    """
    Plot the cutoff lines to each plot

    :Arguments:
        :type cut_S: pandas.Series
        :param cut_S: contains a cutoff value, name and color

        :type ax: matplotlib.axes._subplots.AxesSubplot
        :param ax: Gets an ax project.

        :type p: float
        :param p: percentile of cutoff
    """
    lines.drawCutoffHoriz(ax=ax,y=float(cut_S.values[0]),
            cl=cutPalette.ugColors[cut_S.name],
            lb="{0} {1}% Threshold: {2}".format(cut_S.name,round(p*100,3),
            round(float(cut_S.values[0]),1)),ls="--",lw=2)


def makePlots (SEDData, design, pdf, groupName, cutoff, p, plotType, ugColors, levels):
    """
    Manage all the plots for this script

    :Arguments:
        :type SEDData: pandas.dataFrame
        :param SEDData: Contains SED data either to Mean or pairwise

        :type design: pandas.dataFrame
        :param design: Design file after getColor

        :type pdf: PDF object
        :param pdf: PDF for output plots

        :type groupName: string
        :param groupName: Name of the group (figure title).

        :type cutoff: pandas.dataFrame
        :param cutoff: Cutoff values, beta, chi-sqr and normal.

        :type p: float
        :param p: Percentil for cutoff.

        :type plotType: string
        :param plotType: Type of plot, the possible types are scatterplot to mean
            scatterplot pairwise and boxplot pairwise.

    """

    #Geting number of features in dataframe
    nFeatures = len(SEDData.index)

    #Calculates the widht for the figure base on the number of features
    figWidth = max(nFeatures/2, 16)

    # Create figure object with a single axis and initiate the figss
    figure = figureHandler(proj='2d', figsize=(figWidth, 8))

    # Keeping the order on the colors
    SEDData["colors"]=design["colors"]

    # Choose type of plot
    # Plot scatterplot to mean
    if(plotType=="scatterToMean"):
        #Adds Figure title, x axis limits and set the xticks
        figure.formatAxis(figTitle="Standardized Euclidean Distance from samples {} to the mean".
                        format(groupName),xlim=(-0.5,-0.5+nFeatures),ylim="ignore",
                        xticks=SEDData.index.values,xTitle="Index",
                        yTitle="Standardized Euclidean Distance")

        #Plot scatterplot quickplot
        scatter.scatter2D(ax=figure.ax[0],colorList=SEDData["colors"],
                        x=range(len(SEDData.index)), y=SEDData["SED_to_Mean"])

    #Plot scatterplot pairwise
    elif(plotType=="scatterPairwise"):
        # Adds Figure title, x axis limits and set the xticks
        figure.formatAxis(figTitle="Pairwise standardized Euclidean Distance from samples {}".
                        format(groupName),xlim=(-0.5,-0.5+nFeatures),ylim="ignore",
                        xticks=SEDData.index.values,xTitle="Index",
                        yTitle="Standardized Euclidean Distance")

        # Plot scatterplot
        for index in SEDData.index.values:
            scatter.scatter2D(ax=figure.ax[0],colorList=design["colors"][index],
                            x=range(len(SEDData.index)), y=SEDData[index])

    #Plot boxplot pairwise
    elif(plotType=="boxplotPairwise"):
        # Add Figure title, x axis limits and set the xticks
        figure.formatAxis(figTitle="Box-plots for pairwise standardized Euclidean Distance from samples {}".
                        format(groupName),xlim=(-0.5,-0.5+nFeatures),ylim="ignore",
                        xticks=SEDData.index.values,xTitle="Index",
                        yTitle="Standardized Euclidean Distance")
        # Plot Box plot
        box.boxDF(ax=figure.ax[0], colors=SEDData["colors"].values, dat=SEDData)

    #Add a cutoof line
    cutoff.apply(lambda x: plotCutoffs(x,ax=figure.ax[0],p=p),axis=0)
    figure.shrink()
    # Plot legend
    figure.makeLegend(figure.ax[0], ugColors, levels)
    # Add figure to PDF and close the figure afterwards
    figure.addToPdf(pdf)


def prepareSED(data_df, design, pdf, groupName, p, ugColors, levels):
    """
    Core for processing all the data.

    :Arguments:
        :type data_df: pandas.Series.
        :param data_df: contains a cutoff value, name and color.

        :type design: matplotlib.axes._subplots.AxesSubplot.
        :param design: Gets an ax project.

        :type pdf: PDF object.
        :param pdf: PDF for output plots.

        :type groupName: string.
        :param groupName: Name of the group (figure title).

        :type p: float.
        :param p: percentile of cutoff.

    :Returns:
        :rtype pdf: PDF object.
        :return pdf: PDF for output plots.

        :rtype SEDtoMean: pd.DataFrames
        :return SEDtoMean: SEd for Mean

        :rtype SEDpairwise: pd.DataFrames
        :return SEDpairwise: SED for pairwise data
    """
    # Calculate SED without groups
    SEDtoMean, SEDpairwise = getSED(data_df)

    # Calculate cutOffs
    cutoff1,cutoff2 = getCutOffs(data_df, p)

    # Call function to do a scatter plot on SEDs from samples to the Mean
    makePlots(SEDtoMean, design, pdf, groupName, cutoff1, p, "scatterToMean",
                ugColors, levels)

    #Call function to do a scatter plot on SEDs for pairwise samples
    makePlots(SEDpairwise, design, pdf, groupName, cutoff2, p, "scatterPairwise",
                ugColors, levels)

    # Call function to do a boxplot on SEDs for pairwise samples
    makePlots(SEDpairwise, design, pdf, groupName, cutoff2, p, "boxplotPairwise",
                ugColors, levels)

    #Returning data
    return pdf, SEDtoMean, SEDpairwise


def calculateSED(dat, levels, combName, pdf, p):
    """
    Manage all the plots for this script

    :Arguments:
        :type dat: wideToDesign
        :param dat: Contains data parsed by interface

        :type args: argparse.data
        :param args: All input data

        :type levels: string
        :param levels: Name of the column on desing file (after get colors)
                         with the name of the column containing the combinations.

        :type combName: dictionary
        :param combName: dictionary with colors and different groups
    """

    if len(levels.keys()) > 1:
        # Create dataframes for later use for SED results
        SEDtoMean=pd.DataFrame(columns=['SampleID','SED_to_Mean', 'group'])
        SEDpairwise=pd.DataFrame(columns=['SampleID', 'group'])

        # Calculate pairwise and mean distances by group or by levels
        for level, group in dat.design.groupby(combName):
            # Subset wide
            currentFrame = dat.wide[group.index]

            # Log an error if there are fewer than 3 groups
            if len(group.index) < 3:
                logger.error("Group {0} has less than 3 elements".\
                    format(level))
                exit()

            # Getting SED per group
            logger.info("Getting SED for {0}".format(level))
            pdf, SEDtoMean_G, SEDpairwise_G = prepareSED(currentFrame, group,
                                                pdf, "in group "+str(level), p,
                                                levels, combName)

            # Add 'group' column to the current group
            SEDtoMean_G['group'] = [level]*len(currentFrame.columns)
            SEDtoMean_G['SampleID'] = SEDtoMean_G.index
            SEDpairwise_G['group'] = [level]*len(currentFrame.columns)
            SEDpairwise_G['SampleID'] = SEDpairwise_G.index

            # Merge group dataframes into the all-encompassing dataframe
            SEDtoMean = pd.DataFrame.merge(SEDtoMean,
                                            SEDtoMean_G,
                                            on=['SED_to_Mean', 'group', 'SampleID'],
                                            how='outer',
                                            sort=False)


            SEDpairwise = pd.DataFrame.merge(SEDpairwise,
                                            SEDpairwise_G,
                                            on=['group', 'SampleID'],
                                            how='outer',
                                            sort=False)

        # Recover the index from SampleID
        SEDtoMean.set_index('SampleID', inplace=True)
        SEDpairwise.set_index('SampleID', inplace=True)

        # Get means for all different groups
        logger.info("Getting SED for all data")
        cutoffAllMean,cutoffAllPairwise = getCutOffs(dat.wide,p)

        # Sort df by group
        SEDtoMean = SEDtoMean.sort_values(by='group')
        SEDpairwise = SEDpairwise.sort_values(by='group')

        # Plot a scatter plot on SEDs from samples to the Mean
        makePlots (SEDtoMean, dat.design, pdf, "", cutoffAllMean, p,
                    "scatterToMean", levels, combName)

        # Plot a scatter plot on SEDs for pairwise samples
        makePlots (SEDpairwise, dat.design, pdf, "", cutoffAllPairwise, p,
                    "scatterPairwise", levels, combName)

        # Plot a boxplot on SEDs for pairwise samples
        makePlots (SEDpairwise, dat.design, pdf, "", cutoffAllPairwise, p,
                    "boxplotPairwise", levels, combName)

        # If group drop "group" column
        SEDtoMean.drop('group', axis=1, inplace=True)
        SEDpairwise.drop('group', axis=1, inplace=True)

    else:
        logger.info("Getting SED for all data")
        pdf,SEDtoMean,SEDpairwise = prepareSED(dat.wide, dat.design, pdf,'', p,
                                        levels, combName)

    return SEDtoMean,SEDpairwise


def getSED(wide):
    """
    Calculate the Standardized Euclidean Distance and return an array of
    distances to the Mean and a matrix of pairwise distances.

    :Arguments:
        :type wide: pandas.DataFrame
        :param wide: A wide formatted data frame with samples as columns and
                     compounds as rows.

    :Returns:
        :return: Return 4 pd.DataFrames with SED values and cutoffs.
        :rtype: pd.DataFrames
    """
    # Calculate means
    mean = pd.DataFrame(wide.mean(axis=1))

    # Calculate variance
    variance = wide.var(axis=1,ddof=1)

    # Flag if variance == 0
    variance[variance==0]=1

    # Get SED distances!
    dist = DistanceMetric.get_metric('seuclidean', V=variance)

    # Calculate the SED from all samples to the mean
    SEDtoMean = dist.pairwise(wide.values.T, mean.T)
    SEDtoMean = pd.DataFrame(SEDtoMean, columns = ['SED_to_Mean'],
                               index = wide.columns)

    # Calculate the pairwise standardized Euclidean Distance of all samples
    SEDpairwise = dist.pairwise(wide.values.T)
    SEDpairwise = pd.DataFrame(SEDpairwise, columns=wide.columns,
                               index=wide.columns)

    # Convert the diagonal to NaN
    for index, row in SEDpairwise.iterrows():
        SEDpairwise.loc[index, index] = np.nan

    #Returning data
    return SEDtoMean,SEDpairwise


def getCutOffs(wide,p):
    """
    Calculate the Standardized Euclidean Distance and return an array of
    distances to the Mean and a matrix of pairwise distances.

    :Arguments:
        :type wide: pandas.DataFrame
        :param wide: A wide formatted data frame with samples as columns and
                     compounds as rows.

        :type p: float.
        :param p: percentile of cutoff.

    :Returns:
        :rtype cutoff1: pandas.dataFrame
        :return cutoff1: Cutoff values for mean, beta, chi-sqr and normal.

        :rtype cutoff2: pandas.dataFrame
        :return cutoff2: Cutoff values for pairwise, beta, chi-sqr and normal.
    """

    # Establish iterations, and numer of colums ps and number of rows nf
    ps  = len(wide.columns)
    nf = len(wide.index)
    iters  = 20000

    # Calculate betaP
    betaP=np.percentile(pd.DataFrame(stats.beta.rvs(0.5, 0.5*(ps-2),size=iters*nf).reshape(iters,nf)).sum(axis=1), p*100.0)

    #casting to float so it behaves well
    ps = float(ps)
    nf = float(nf)

    # Calculate cutoffs beta, norm, and chisq for data to mean
    betaCut1  = np.sqrt((ps-1)**2/ps*betaP)
    normCut1  = np.sqrt(stats.norm.ppf(p, (ps-1)/ps*nf,
                        np.sqrt(2*nf*(ps-2)*(ps-1)**2/ps**2/(ps+1))))
    chisqCut1 = np.sqrt((ps-1)/ps*stats.chi2.ppf(p, nf))

    # Calculate cutoffs beta,n norm & chisq for pairwise
    betaCut2  = np.sqrt((ps-1)*2*betaP)
    normCut2  = np.sqrt(stats.norm.ppf(p, 2*nf, np.sqrt(8*nf*(ps-2)/(ps+1))))
    chisqCut2 = np.sqrt(2*stats.chi2.ppf(p, nf))

    #Create data fram for ecah set of cut offs
    cutoff1   = pd.DataFrame([[betaCut1, normCut1, chisqCut1],
                            ['Beta(Exact)', 'Normal', 'Chi-sq']],index=["cut","name"],
                            columns=['Beta(Exact)', 'Normal', 'Chi-sq'])
    cutoff2   = pd.DataFrame([[betaCut2, normCut2, chisqCut2],
                            ['Beta(Exact)', 'Normal', 'Chi-sq']],index=["cut","name"],
                            columns=['Beta(Exact)', 'Normal', 'Chi-sq'])

    # Create a palette
    cutPalette.getColors(cutoff1.T,["name"])

    # Return colors
    return cutoff1,cutoff2


def main(args):
    """
    Main function
    """
    #Getting palettes for data and cutoffs
    global cutPalette
    cutPalette = ch.colorHandler(pal="tableau",col="TrafficLight_9")

    # Checking if levels
    if args.levels and args.group:
        levels = [args.group]+args.levels
    elif args.group and not args.levels:
        levels = [args.group]
    else:
        levels = []

    #Parsing data with interface
    dat = wideToDesign(args.input, args.design, args.uniqID, group=args.group,
                        anno=args.levels,  logger=logger, runOrder=args.order)

    #Dropping missing values and remove groups with just one sample
    dat.dropMissing()
    if args.group:
        dat.removeSingle()

    #Select colors for data
    dataPalette.getColors(design=dat.design, groups=levels)
    dat.design=dataPalette.design

    #Open pdfPages Calculate SED
    with PdfPages(os.path.abspath(args.figure)) as pdf:
        SEDtoMean,SEDpairwise=calculateSED(dat, dataPalette.ugColors, dataPalette.combName, pdf, args.p)


    #Outputing files for tsv files
    SEDtoMean.to_csv(os.path.abspath(args.toMean), index_label="sampleID",
                    columns=["SED_to_Mean"],sep='\t')
    SEDpairwise.drop(["colors"],axis=1,inplace=True)
    if args.group:
        SEDpairwise.drop(["colors_x","colors_y"],axis=1,inplace=True)
    SEDpairwise.to_csv(os.path.abspath(args.pairwise),index_label="sampleID",
                    sep='\t')

    #Ending script
    logger.info("Script complete.")


if __name__ == '__main__':
    # Turn on Logging if option -g was given
    args = getOptions()
    logger = logging.getLogger()
    sl.setLogger(logger)
    logger.info("Importing data with following parameters: "\
                "\n\tWide: {0}"\
                "\n\tDesign: {1}"\
                "\n\tUnique ID: {2}"\
                "\n\tGroup: {3}"\
                "\n\tRun Order: {4}".format(args.input, args.design,
                 args.uniqID, args.group, args.order))
    dataPalette = colorHandler(pal=args.palette, col=args.color)
    logger.info(u"Using {0} color scheme from {1} palette".format(args.color,
                args.palette))
    main(args)
