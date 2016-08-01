#!/usr/bin/env python
######################################################################################
# Date: 2016/July/06
# 
# Module: peak_compare.py
#
# VERSION: 1.0
# 
# AUTHOR: Miguel Ibarra (miguelib@ufl.edu)
#
# DESCRIPTION: This program compares the features among 2 annotation files based
#				on their retention time and mz.It will output the results of this
#				comparison at level of combinations and features.
#
#######################################################################################

#Standard Libraries
import argparse
import os
import logging
import itertools

#AddOn Libraries
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

#Local Libraries
import interface
import logger as sl
import module_venn as mVenn

#Getting all the arguments
def getOptions():
    """Function to pull arguments"""
    parser = argparse.ArgumentParser(description="""Matches rows (features) in 2
                                     files by their m/z and RT values""")
    
    required = parser.add_argument_group(title='Required Input', 
                                    description='Required input to the program')
    required.add_argument('-a1',"--anno1",dest="anno1", action="store",
                        required=True,help="Out path for first file")
    required.add_argument('-a2',"--anno2",dest="anno2", action="store",
                        required=True,help="Out path for second file")
    required.add_argument("-ID1","--uniqID1",dest="uniqID1",action="store",
                        required=True, default = "rowID",help="""Name of the 
                        column in file1 that contains the uniqID""")
    required.add_argument("-mz1","--mzID1",dest="mzID1",action="store",
                        required=True, default = "RT",help="""Name of the column
                         in file1 that contains MZ""")
    required.add_argument("-rt1","--rtID1",dest="rtID1",action="store",
                        required=True, default = "MZ",help="""Name of the column
                         in file1 that contains RT""")
    required.add_argument("-ID2","--uniqID2",dest="uniqID2",action="store",
                        required=True, default = "rowID",help="""Name of the 
                        column in file2 that contains the uniqID""")
    required.add_argument("-mz2","--mzID2",dest="mzID2",action="store",
                        required=True,default = "MZ",help="""Name of the column
                         in file2 that contains MZ""")
    required.add_argument("-rt2","--rtID2",dest="rtID2",action="store",
                        required=True,default = "RT",help="""Name of the column
                         in file2 that contains RT""")

    output = parser.add_argument_group(title='Output files', 
                                    description='Output paths for the program')
    output.add_argument('-a', "--all",  dest="all", action='store',
                        required=True,help="Out path for All peak combinations File")
    output.add_argument('-m', "--matched", dest="matched", action='store',
                        required=True,help="Out path for Matched peaks combinations File")
    output.add_argument('-u1', "--unmatched1", dest="unmatched1", action='store',
                        required=True,help="Out path for Unmatched peaks in file 1")
    output.add_argument('-u2',"--unmatched2", dest="unmatched2", action='store',
                        required=True,help="Out path for Unmatched peaks in file 2")
    output.add_argument('-s',"--summary", dest="summary", action='store',
                        required=True,help="Out path for Summary File")
    output.add_argument('-fig',"--figure", dest="figure", action='store',
                        required=True,help="""Out path for Matched vs Unmatched 
                        Combinations Venn Diagram File""")
    
    optional = parser.add_argument_group(title="Optional Input", 
                                    description="Optional Input to the program")
    optional.add_argument('-mz',"--mzcut",dest="mzcut",action='store',
                        required=False, default="0.005",help="""Window value for 
                        MZ matching [default 0.005]""")
    optional.add_argument('-rt',"--rtcut",dest="rtcut",action='store',
                        required=False, default="0.15",help="""Window value for 
                        RT matching [default 0.15]""")
    optional.add_argument('-n1',"--name1",dest="name1",action='store',
                        required=False, default="F1",help="""Short name for File 1
                         [default F1]""")
    optional.add_argument('-n2',"--name2",dest="name2",action='store',
                        required=False, default="F2",help="""Short name for File 2
                         [default F2]""")
    optional.add_argument('-g',"--log",dest="LOG",action="store",
                        required=False,help="Path and name of the log file")

    args = parser.parse_args()
    return(args);
 
def matchFiles (anno1,anno2,MZCut,RTCut,reverse=False):
    """ 
    Match 2 Files and returns an array with the results.

    :Arguments:
        :type anno1: pd.DataFrame.
        :param anno1: Dataframe with the annotation file 1.

        :type anno2: pd.DataFrame.
        :param anno2: Dataframe with the annotation file 2.

        :type MZCut: int
        :param MZCut: window size for mz.

        :type RtCut: int
        :param RtCut: window size for rt.

        :type reverse: boolean.
        :param reverse: flag to determinte wich way the match is going to be done.

    :Returns:
        :return: Returns a figure object.
        :rtype: matplotlib.pyplot.Figure.figure

    """
    if reverse:
        logger.info(u"Matching annotation file  1 to annotation file 2")
    else:
        logger.info(u"Matching annotation file  2 to annotation file 1")

    #Creating dataframe for matched 
    matched_df = pd.DataFrame(columns=["rowID1","MZ1","RT1","rowID2","MZ2","RT2"])
    unmatched_df = pd.DataFrame(columns=["rowID1","MZ1","RT1","rowID2","MZ2","RT2"])

    #Iterating over annotation 1 and comparing with annotation 2
    for rowID1,MZRT1 in anno1.anno.iterrows():

        #Setting  flag_unmatch
        flag_unmatch = True

        #Calculating MZ and RT cuts
        mzMin = MZRT1[anno1.mz] - MZCut
        mzMax = MZRT1[anno1.mz] + MZCut
        rtMin = MZRT1[anno1.rt] - RTCut
        rtMax = MZRT1[anno1.rt] + RTCut

        #Iterating over annotation 2 and comparing with anotation 1
        for rowID2,MZRT2 in anno2.anno.iterrows():

                #Match found between anno1 and anno2
                if ((mzMin<MZRT2[anno2.mz] and MZRT2[anno2.mz]<mzMax) and 
                    (rtMin<MZRT2[anno2.rt] and MZRT2[anno2.rt]<rtMax)):

                    #If reverse reverse the output 
                    if reverse:
                        matched_s = pd.DataFrame(data=[[rowID2,MZRT2[anno2.mz],
                            MZRT2[anno2.rt],rowID1,MZRT1[anno1.mz],MZRT1[anno1.rt]]],
                            columns=["rowID1","MZ1","RT1","rowID2","MZ2","RT2"])

                    #Creting dataframe to apend latter
                    else:
                        matched_s = pd.DataFrame(data=[[rowID1,MZRT1[anno1.mz],
                            MZRT1[anno1.rt],rowID2,MZRT2[anno2.mz],MZRT2[anno2.rt]]],
                            columns=["rowID1","MZ1","RT1","rowID2","MZ2","RT2"])

                    matched_df = matched_df.append(matched_s)
                    flag_unmatch = False

        #Exclusively found on anno1
        if flag_unmatch:

            #if reverse reverse output
            if reverse:
                unmatched_s = pd.DataFrame(data=[["","","",rowID1,MZRT1[anno1.mz],
                    MZRT1[anno1.rt]]], columns=["rowID1","MZ1","RT1",
                    "rowID2","MZ2","RT2"])

            #Create dataframe to append to unmateched dataframe
            else:
                unmatched_s = pd.DataFrame(data=[[rowID1,MZRT1[anno1.mz],
                    MZRT1[anno1.rt],"","",""]], columns=["rowID1","MZ1","RT1",
                    "rowID2","MZ2","RT2"])
            unmatched_df = unmatched_df.append(unmatched_s)

    #Returning results
    return matched_df,unmatched_df

def getSummary (match,umatch1,umatch2):
    """ 
    Plot the standardized Euclidean distance plot for samples to the Mean.

    :Arguments:
        :type match: pd.DataFrame
        :param match: Dataframe with just the matched combinations

        :type umatch1: pd.DataFrame
        :param umatch1: Dataframe with the unmatched combinations for file 1.

        :type umatch2: pd.DataFrame
        :param umatch2: Dataframe with the unmatched combinations for file 2.

    :Returns:
        :return: dictionary.
        :rtype: dictionary with all the summary information.

    """

    logger.info(u"Summarizing results")

    #Calculate combinations
    nUmatchCombinations1 = len(umatch1) #This is the number of unmatched features
    nUmatchCombinations2 = len(umatch2) #This is the number of unmatched features
    nMatchCombinations = len(match)
    nAllCombinations = len(match) + len(umatch1) + len(umatch2)
    
    #Calculate number of features
    nMatchFeatures1 = len(list(set(match["rowID1"].values)))
    nMatchFeatures2 = len(list(set(match["rowID2"].values)))
    nMatchFeatures = nMatchFeatures1 + nMatchFeatures2
    nAllFeatures = nUmatchCombinations1+nUmatchCombinations2+nMatchFeatures
    
    #Calculate number of multiple features
    nMultupleFeatures1 = len([len(list(count)) for name,count in 
        itertools.groupby(sorted(match["rowID1"])) if (len(list(count))>1)])
    nMultipleFeatures2 = len([len(list(count)) for name,count in 
        itertools.groupby(sorted(match["rowID2"])) if (len(list(count))>1)])
    nMultipleFeatures = nMultupleFeatures1+nMultipleFeatures2
    
    #Calculate number of single features
    nSingleFeatures1 = nMatchFeatures1-nMultupleFeatures1
    nSingleFeatures2 = nMatchFeatures2-nMultipleFeatures2
    nSingleFeatures = nSingleFeatures1+nSingleFeatures2
    
    #Creating series 
    summary_S = pd.Series([nUmatchCombinations1,nUmatchCombinations2,
                           nMatchCombinations,nAllCombinations,
                           nMatchFeatures1,nMatchFeatures2,
                           nMatchFeatures,nAllFeatures,
                           nMultupleFeatures1,nMultipleFeatures2,
                           nMultipleFeatures,nSingleFeatures1,
                           nSingleFeatures2,nSingleFeatures],
                          index=["UmatchCombinations1","UmatchCombinations2",
                                "MatchCombinations","AllCombinations",
                                "MatchFeatures1","MatchFeatures2",
                                "MatchFeatures","AllFeatures",
                                "MultupleFeatures1","MultipleFeatures2",
                                "MultipleFeatures","SingleFeatures1",
                                "SingleFeatures2","SingleFeatures"])
    return summary_S

def plotFigures(args,summary):
    """ 
	Plot the Venn diagrams of the combinations and features

    :Arguments:
        :type args: argparse object
        :param args: argparse object with all the input parameters, from here 
        			we are just going to take the short names and the out paths.

        :type summary: dictionary
        :param summary: dictionary with all the summary information.
	"""
    logger.info(u"Plotting Figures")

    with PdfPages(os.path.abspath(args.figure)) as pdfOut:
        #Venn match unmatch combinations
        mvsumCombFig=mVenn.plotVenn2([summary["UmatchCombinations1"],
            summary["UmatchCombinations2"],summary["MatchCombinations"]],
            title="MZ-RT Matched vs Unmatched (Combinations)",name1=args.name1,
            name2=args.name2,circles=True)
        mvsumCombFig.addToPdf(dpi=600, pdfPages=pdfOut)

        #Venn match vs unmatch features
        mvsumFeatFig=mVenn.plotVenn2([summary["UmatchCombinations1"],
            summary["UmatchCombinations2"],summary["MatchFeatures"]],
            title="MZ-RT Matched vs Unmatched (Features)",name1=args.name1,
            name2=args.name2,circles=True)
        mvsumFeatFig.addToPdf(dpi=600, pdfPages=pdfOut)

        #Venn single vs multiple
        svsuFeatFig=mVenn.plotVenn2([summary["SingleFeatures1"]
            ,summary["SingleFeatures2"],summary["MultipleFeatures"]],
            title="MZ-RT Single vs Multiple",name1=args.name1,
            name2=args.name2,circles=True)
        svsuFeatFig.addToPdf(dpi=600, pdfPages=pdfOut)

def writeOutput(paths,files):
    """
    Writes output

    :Arguments:
        :type paths: list
        :param paths: list of out paths

        :type files: list
        :param files: list of pandas data frames
    """

    #For each dataframe saves it in its specific file.
    #File names must match!
    logger.info(u"Writting output")
    for i in range(len(files)):
        files[i].to_csv(paths[i],index=False,sep="\t")

def main(args):
    """Function to call all other functions"""
    #Read annotation file 1
    anno1 = interface.annoFormat(anno=args.anno1, uniqID=args.uniqID1, 
                                mz=args.mzID1, rt=args.rtID1)

    #Read annotation file 1
    anno2 = interface.annoFormat(anno=args.anno2, uniqID=args.uniqID2, 
                                mz=args.mzID2, rt=args.rtID2)

    #Matching files anno1 vs anno2
    match12_df,umatch12_df = matchFiles(anno1=anno1,anno2=anno2,MZCut=float(args.mzcut),
        RTCut=float(args.rtcut))

    #Matching files anno1 vs anno2
    match21_df,umatch21_df = matchFiles(anno1=anno2,anno2=anno1,MZCut=float(args.mzcut),
        RTCut=float(args.rtcut), reverse = True)

    #concatenate match results
    match_df = pd.concat([match12_df,match21_df],axis=0)

    #Remove duplicates from match
    match_df.drop_duplicates(inplace=True)

    #Create all results file
    all_df = pd.concat([match_df,umatch12_df,umatch21_df],axis=0)

    #get summary data
    summary_S = getSummary(match_df,umatch12_df,umatch21_df)

    #Plot venn Diagrams
    plotFigures(args,summary_S)

    #Output data
    writeOutput(paths=[args.unmatched1,args.unmatched2,args.matched,args.all],
                files=[umatch12_df,umatch21_df,match_df,all_df])

    summary_S.to_csv(args.summary,sep="\t")

    logger.info(u"MZERMatching finalized successfully")

if __name__=='__main__':
    #Get info
    args = getOptions()

    #Setting logger
    logger = logging.getLogger()
    sl.setLogger(logger)

    # Showing parameters
    logger.info(u"""Importing data with the following parameters:
        \tAnnotation 1: {0}
        \tAnnotation 2: {1}
        """.format(args.anno1,args.anno2))

    # Runing script
    main(args)
