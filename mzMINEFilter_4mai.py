#!/usr/bin/env python
#Standard Libraries
import argparse
import os
import copy
import math
import logging

#AddOn Libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def getOptions():
    """Function to pull arguments"""
    parser = argparse.ArgumentParser(description="Takes a mzMINE file format csv and filter/tags the rows using an LOD score")
    parser.add_argument("-sf", dest="sybsfile", action='store', required=True, help="Name of mzMINE file [required]")
    parser.add_argument("-df", dest="dfile", action='store', required=True, help="Name of mzMINE file [required]")
    parser.add_argument("-o", dest="oname", action='store', required=True, help="Name of the output cvs format file [required]")
    parser.add_argument("--odir", dest="odir", action='store', required=True,help="Output directory path [required]")
    parser.add_argument("-L", dest="lod", action='store', required=False, default=5000, help="LOD Value default [default 5000]")
    parser.add_argument("-c", dest="criteria", action='store', required=False, default=10, help="Threshold value [default 10]")
    parser.add_argument("-lg", dest="LODgroup", action='store', required=False, default='blank', help="Group to calculate the LOD value [default 'blank']")
    parser.add_argument("-g", "--log", dest="log", action='store', required=False, default=False, help="Path and name of log file")
    args = parser.parse_args()
    #args = parser.parse_args(['-sf', 'test_data/mcdaniels/NE_RT_sbys_test.tsv', '-df', 'test_data/mcdaniels/design_test.tsv', '--odir', 'test_data/mcdaniels', '-o', 'results.tsv', '-L', '5.1', '-c', '0', '-lg', 'b'])
    return (args)

def readDesignFile(designPath):
    """Function to read Design Files"""
    bname = os.path.basename(designPath)
    name,format = os.path.splitext(bname)

    if args.log: logger.info("Reading design file '%s' " % (designPath))
    if format == '.tsv':
        designFile = np.genfromtxt(fname=designPath, dtype='str', delimiter='\t')
    elif format == '.csv':
        designFile = np.genfromtxt(fname=designPath, dtype='str', delimiter=',')

    groupIndex = list(designFile[0]).index('group')
    sampleIDIndex = list(designFile[0]).index('sampleID')

    SampleID_group = {designFile[i][sampleIDIndex]:designFile[i][groupIndex] for i in range(len(designFile)) if i!=0}
    Groups = list   (set([designFile[i][groupIndex]for i in range(len(designFile)) if i!=0]))
    if args.log:logger.info("Finished reading design File '%s' " %(designPath))
    return SampleID_group,Groups

def readSBYSFile (sbysPath,SampleID_group,Groups):
    """Function to read Side by Side Files"""
    bname = os.path.basename(sbysPath)
    name,format = os.path.splitext(bname)

    if args.log:logger.info("Reading Side by Side File '%s' " %(sbysPath))
    if format == '.tsv':
        SBYSFile = np.genfromtxt(fname=sbysPath, dtype='str', delimiter='\t')
    elif format == '.csv':
        SBYSFile = np.genfromtxt(fname=sbysPath, dtype='str', delimiter=',')

    SBYSRowID_Group_Values = {row[0]:{group:[] for group in Groups} for row in SBYSFile if not(row[0].startswith('rowID'))}
    for row in SBYSFile[1:]:
        for i in range(len(row)):
            if i==0:continue
            SBYSRowID_Group_Values[row[0]][SampleID_group[SBYSFile[0][i]]].append(row[i])
    if args.log:logger.info("Finished reading Side by Side File '%s' " %(sbysPath))
    return (SBYSRowID_Group_Values)

def getLODValues (SBYSRowID_Group_Values, LOD, LODgroup):
    """Function to obtain de LOD values per feature"""
    if args.log:logger.info("Calculating LOD values")
    rowID_LODValue = {}
    for rowID in SBYSRowID_Group_Values:
        for  group in SBYSRowID_Group_Values[rowID]:
            if group==LODgroup:
                featureBlanks = SBYSRowID_Group_Values[rowID][LODgroup]
                featureBlanks = map(float,featureBlanks)
                rowID_LODValue[rowID] = round(np.average(featureBlanks)+(3*np.std(featureBlanks,ddof=1)),3)
                if rowID_LODValue[rowID] == 0:
                    rowID_LODValue[rowID] = LOD
    if args.log:logger.info("Finished calculating LOD values")
    return rowID_LODValue

def getmzMINEMeans (SBYSRowID_Group_Values,Groups,LODgroup):
    """Function to obtain the means per feature"""
    if args.log:logger.info("Calculating group means")
    rowID_group_mean = {rowID:{group:0 for group in Groups if group!=LODgroup} for rowID in SBYSRowID_Group_Values}
    for rowID in SBYSRowID_Group_Values:
        for group in SBYSRowID_Group_Values[rowID]:
            if group != LODgroup:
                GroupValues = SBYSRowID_Group_Values[rowID][group]
                GroupValues = map(float,GroupValues)
                featureMean = round(np.average(GroupValues),3)
                rowID_group_mean[rowID][group] = featureMean
    if args.log:logger.info("Finished calculating group means")
    return rowID_group_mean

def getFiltersmzMINE (rowID_LODValue,rowID_group_mean,Groups,LODgroup):
    """Function to obtain the filtered values"""
    if args.log:logger.info("Calculating group Filters")
    rowID_group_filter = {rowID:{group:0 for group in Groups if group!=LODgroup} for rowID in rowID_group_mean}
    for rowID in rowID_group_mean:
        for group in rowID_group_mean[rowID]:
            rowID_group_filter[rowID][group] = round(((rowID_group_mean[rowID][group]-rowID_LODValue[rowID])/rowID_LODValue[rowID]),3)
    if args.log:logger.info("Finished calculating group Filters")
    return rowID_group_filter

def getFlagsmzMINE (rowID_group_filter, criteria,Groups,LODgroup):
    """Function to set the flags depending on the criteria and the filtered values"""
    if args.log:logger.info("Calculating feature flags")
    rowID_group_flag = {rowID:{group:0 for group in Groups if group!=LODgroup} for rowID in rowID_group_filter}
    for rowID in rowID_group_filter:
        rowID_group_flag[rowID]["flag_LODFilter_all_off"] = 0
        rowID_group_flag[rowID]["flag_LODFilter_atleastone_off"] = 0
        countflags = 0
        for group in rowID_group_filter[rowID]:
            if rowID_group_filter[rowID][group] > criteria:
                rowID_group_flag[rowID][group] = 0
            else:
                rowID_group_flag[rowID][group] = 1
                countflags += 1
        if countflags > 0:
            rowID_group_flag[rowID]["flag_LODFilter_atleastone_off"] = 1
        if countflags == len(Groups)-1:
            rowID_group_flag[rowID]["flag_LODFilter_all_off"] = 1
    if args.log:logger.info("Finished calculating feature flags")
    return rowID_group_flag

def writeOut (rowID_group_mean,rowID_group_filter,rowID_group_flag,Groups,LODgroup,odirPath,outFilePath):
    """Function to write tables"""
    outfile = os.path.basename(outFilePath)
    outfile = os.path.splitext(outfile)[0]
    GroupNames = [group for group in Groups if group != LODgroup]
    GroupNames.insert(0,'rowID')

    if args.log:logger.info("Printing means file to" '%s'%(os.path.join(odirPath,outfile+'_means.tsv')))
    OUTFILE = open(os.path.join(odirPath,outfile+'_means.tsv'),'w')
    print >> OUTFILE,'\t'.join(GroupNames)
    for rowID in rowID_group_mean:
        rowMeans = [str(rowID_group_mean[rowID][group]) for group in Groups if group != LODgroup]
        rowMeans.insert(0,rowID)
        print >> OUTFILE,'\t'.join(rowMeans)
    OUTFILE.close()
    if args.log:logger.info("Finished printing means file to" '%s'%(os.path.join(odirPath,outfile+'_means.tsv')))

    if args.log:logger.info("Printing filters file to" '%s'%(os.path.join(odirPath,outfile+'_LODfilters.tsv')))
    OUTFILE = open(os.path.join(odirPath,outfile+'_LODfilters.tsv'),'w')
    print >> OUTFILE,'\t'.join(GroupNames)
    for rowID in rowID_group_filter:
        rowFilters = [str(rowID_group_filter[rowID][group]) for group in Groups if group != LODgroup]
        rowFilters.insert(0,rowID)
        print >> OUTFILE,'\t'.join(rowFilters)
    OUTFILE.close()
    if args.log:logger.info("Finished printing filters file to" '%s'%(os.path.join(odirPath,outfile+'_LODfilters.tsv')))

    if args.log:logger.info("Printing flags file to" '%s' % (os.path.join(odirPath,outfile+'_Flags.tsv')))
    OUTFILE = open(os.path.join(odirPath,outfile+'_Flags.tsv'),'w')
    GroupNames.append('at_least_one')
    GroupNames.append('all')
    print >> OUTFILE,'\t'.join(GroupNames)
    for rowID in rowID_group_flag:
        rowFlags = [str(rowID_group_flag[rowID][group]) for group in Groups if group != LODgroup]
        rowFlags.insert(0,rowID)
        rowFlags.append(str(rowID_group_flag[rowID]["flag_LODFilter_atleastone_off"]))
        rowFlags.append(str(rowID_group_flag[rowID]["flag_LODFilter_all_off"]))
        print >> OUTFILE,'\t'.join(rowFlags)
    OUTFILE.close()
    if args.log:logger.info("finished printing flags file to" '%s' % (os.path.join(odirPath,outfile+'_Flags.tsv')))

def plotHeatMap (rowID_group_mean,Groups,LODgroup,odirPath,outFilePath):
    """Function to plot a HeatMap"""
    outfile = os.path.basename(outFilePath)
    outfile = os.path.splitext(outfile)[0]
    if args.log:logger.info("Plotting HeatMap to" '%s' % (os.path.join(odirPath,outfile+'HeatMap.png')))
    GroupNames = [group for group in Groups if group != LODgroup]

    dataToPlot = []
    rowIDs = []
    for rowID in rowID_group_mean:
        rowMeans = [rowID_group_mean[rowID][group] for group in Groups if group != LODgroup]
        rowIDs.append(rowID)
        dataToPlot.append(rowMeans)

    dataToPlot =  np.array(dataToPlot)#Convert data (list of list) to numpy array 

    fig, ax = plt.subplots(1) #Create figure
    p = ax.pcolormesh(dataToPlot) #Create plot
    fig.colorbar(p) #Plot color bar

    xticks = np.arange(len(GroupNames)) #Set labels
    ax.set_xticks(xticks+0.5) #Positions of the ticks
    ax.set_xticklabels(GroupNames,rotation='horizontal') #Labels of the ticks

    plt.ylim(ymax=len(dataToPlot)) #Set ylim to avoid blank spaces

    #plt.show(fig)
    plt.savefig(os.path.join(odirPath,outfile+'HeatMap.png'), dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)
    if args.log:logger.info("Finished plotting HeatMap to" '%s' % (os.path.join(odirPath,outfile+'HeatMap.png')))

def main(args):
    """Here we call all the other Functions"""
    designFilePath = args.dfile
    sbysFilePath = os.path.abspath(args.sybsfile)
    odirPath =  os.path.abspath(args.odir)
    outFilePath = args.oname
    LOD = float(args.lod)
    LODgroup = args.LODgroup
    criteria = float(args.criteria)

    SampleID_group,Groups = readDesignFile(designFilePath)
    SBYSRowID_Group_Values = readSBYSFile(sbysFilePath,SampleID_group,Groups)
    rowID_LODValue = getLODValues(SBYSRowID_Group_Values,LOD,LODgroup)
    rowID_group_mean = getmzMINEMeans (SBYSRowID_Group_Values,Groups,LODgroup)
    rowID_group_filter = getFiltersmzMINE (rowID_LODValue,rowID_group_mean,Groups,LODgroup)
    rowID_group_flag = getFlagsmzMINE (rowID_group_filter, criteria,Groups,LODgroup)
    writeOut (rowID_group_mean,rowID_group_filter,rowID_group_flag,Groups,LODgroup,odirPath,outFilePath)
    plotHeatMap (rowID_group_mean,Groups,LODgroup,odirPath,outFilePath)

if __name__=='__main__':
    args = getOptions()
    if args.log:
        logging.basicConfig(filename=(os.path.abspath(args.log)),level=logging.DEBUG)
        logger = logging.getLogger()
    main(args)
    if args.log:logger.info("Script complete.")
