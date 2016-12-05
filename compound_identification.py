#!/usr/bin/env python
######################################################################################
# Date: 2016/November/22
# 
# Module: compound_identification.py
#
# VERSION: 0.1
# 
# AUTHOR: Miguel Ibarra (miguelib@ufl.edu)
#
# DESCRIPTION: This program identifies compounds to a local library based on the RT 
#               and m/z.
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

#Getting all the arguments
def getOptions(myopts=False):
    """Function to pull arguments"""
    parser = argparse.ArgumentParser(description="""
    identifies compounds based on a library file""")
    
    required = parser.add_argument_group(title='Required Input', 
                                    description='Required input to the program')
    required.add_argument('-a', "--anno", dest="anno", action="store",
                        required=True, help="Out path for first file")
    required.add_argument("-id", "--uniqID", dest="uniqID", action="store",
                        required=True, default="", help="Name of the"\
                        "column in file that contains the uniqID")
    required.add_argument("-mzi", "--mzID", dest="mzID", action="store",
                        required=True, default="", help="Name of the column"\
                        " in file that contains MZ")
    required.add_argument("-rti", "--rtID", dest="rtID", action="store",
                        required=True, default="", help="Name of the column"\
                        " in file that contains RT")
    required.add_argument('-lib',"--library",dest="library",action='store',
                        required=True, default="",help="Library to use for"\
                        "the matching.")
    required.add_argument("-lid", "--libuniqID", dest="libid", action="store",
                        required=True, default="", help="Name of the"\
                        "column in the library file that contains the uniqID")
    required.add_argument("-lmzi", "--libmzID", dest="libmz", action="store",
                        required=True, default="", help="Name of the column"\
                        " in the library file that contains MZ")
    required.add_argument("-lrti", "--librtID", dest="librt", action="store",
                        required=True, default="", help="Name of the column"\
                        " in the library file that contains RT")

    output = parser.add_argument_group(title='Output files', 
                                    description='Output paths for the program')
    output.add_argument('-o', "--output",  dest="output", action='store',
                        required=True, help="Output path for identified"\
                        "compounds.")

    args = parser.parse_args()
    return(args);
 
def identiyOnTarget (library,target,MZCut,RTCut):
    """ 
    Match 2 Files and returns an array with the results.

    :Arguments:
        :type library: pd.DataFrame.
        :param library: Dataframe with the library to annotate

        :type target: pd.DataFrame.
        :param target: Dataframe with the target of annotation

        :type MZCut: int
        :param MZCut: window size for mz.

        :type RtCut: int
        :param RtCut: window size for rt.

    :Returns:
        :return identified_df: Identified features on target file.
        :rtype identified_df: pd.DataFrame.

    """

    # Creating datafram for identified features 
    #colnames =  target.anno.columns.tolist()
    #colnames.extend([library.uniqID])
    #identified_df = pd.DataFrame(columns=colnames)

    # Iterating over each element on the Target file to assess against library
    identified_feats=list()
    for target_rowID,target_MZRT in target.anno.iterrows():

        # Calculating MZ and RT cuts
        mzMin = target_MZRT[target.mz] - MZCut
        mzMax = target_MZRT[target.mz] + MZCut
        rtMin = target_MZRT[target.rt] - RTCut
        rtMax = target_MZRT[target.rt] + RTCut

        # For those with not idenfied compound imput a blank
        target_MZRT.loc["compound"]=np.nan

        # Iterate over the library file to idenfy compouns/adducts
        for library_rowID, library_MZRT in library.anno.iterrows():

                #Match found between anno1 and anno2
                if ((mzMin<library_MZRT[library.mz] and library_MZRT[library.mz]<mzMax) and 
                    (rtMin<library_MZRT[library.rt] and library_MZRT[library.rt]<rtMax)):        

                    # Creating a series with the data that we want
                    target_MZRT.loc["compound"]=library_rowID            

        # Adding features aggain to identified feats
        identified_feats.append(target_MZRT)

    # Concatenating identified features
    identified_df = pd.concat(identified_feats, axis=1)
    identified_df = identified_df.T
    
    #print identified_df.T["met_356"]
    return identified_df

def main(args):
    # Loading library file
    library = interface.annoFormat(anno=args.library, uniqID=args.libid, 
                                    mz=args.libmz, rt=args.librt)
    # Read target file
    target = interface.annoFormat(anno=args.anno, uniqID=args.uniqID,
                                mz=args.mzID, rt=args.rtID)
                                
    # Matching target file with library file
    logger.info("Identifying compounds")
    identified_df = identiyOnTarget(library=library, target=target, MZCut=0.005,
                     RTCut=0.15)

    # Saving identified compounds to file
    logger.info("Exporting results")
    identified_df.index.name="rowID"
    identified_df.to_csv(os.path.abspath(args.output),sep="\t")

if __name__=='__main__':
    #Get info
    args = getOptions()

    #Setting logger
    logger = logging.getLogger()
    sl.setLogger(logger)

    # Showing parameters
    logger.info(u"""Importing data with following parameters:
            Input:  {0}
            Library:{1}
            """.format(args.anno, args.library))

    # Runing script
    main(args)