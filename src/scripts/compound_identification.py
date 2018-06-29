#!/usr/bin/env python
######################################################################################
# Date: 2016/November/22
# 
# Module: compound_identification.py
#
# VERSION: 0.2
# 
# AUTHOR: Miguel Ibarra (miguelib@ufl.edu)
#
# DESCRIPTION: This program identifies compounds to a local library based on the RT 
#               and m/z.
#
#######################################################################################
# Import built-in libraries
import os
import logging
import argparse
import itertools

# Import add-on libraries
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Import local data ibraries
from secimtools.dataManager import interface
from secimtools.dataManager import logger as sl

#Getting all the arguments
def getOptions(myopts=False):
    """Function to pull arguments"""
    parser = argparse.ArgumentParser(description="""
    Identifies compounds based on a library file""")
    # Requiered Input
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
    # Tool output
    output = parser.add_argument_group(title='Output files', 
                                    description='Output paths for the program')
    output.add_argument('-o', "--output",  dest="output", action='store',
                        required=True, help="Output path for identified"\
                        "compounds.")

    args = parser.parse_args()
    # Stadardize paths
    args.anno    = os.path.abspath(args.anno)
    args.output  = os.path.abspath(args.output)
    args.library = os.path.abspath(args.library)

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
    # Iterating over each element on the Target file to assess against library
    identified_feats=list()
    for target_rowID,target_MZRT in target.data.iterrows():

        # Calculating MZ and RT cuts
        mzMin = target_MZRT[target.mz] - MZCut
        mzMax = target_MZRT[target.mz] + MZCut
        rtMin = target_MZRT[target.rt] - RTCut
        rtMax = target_MZRT[target.rt] + RTCut

        # For those with not idenfied compound imput a blank
        target_MZRT.loc["compound"]=np.nan
        target_MZRT = pd.concat([target_MZRT,pd.Series(index=library.anno)])

        # Iterate over the library file to idenfy compouns/adducts
        for library_rowID, library_MZRT in library.data.iterrows():

                #Match found between anno1 and anno2
                if ((mzMin<library_MZRT[library.mz] and library_MZRT[library.mz]<mzMax) and 
                    (rtMin<library_MZRT[library.rt] and library_MZRT[library.rt]<rtMax)):        

                    # Creating a series with the data that we want
                    target_MZRT.loc["compound"] = library_rowID
                    target_MZRT.loc[library.anno]=library_MZRT[library.anno]

        # Adding features aggain to identified feats
        identified_feats.append(target_MZRT)

    # Concatenating identified features
    identified_df = pd.concat(identified_feats, axis=1)
    identified_df = identified_df.T
    
    # Return identified_df
    return identified_df

def main(args):
    # Loading library and target files
    logger.info("Importing data")
    library = interface.annoFormat(data=args.library, uniqID=args.libid, 
                                    mz=args.libmz, rt=args.librt, anno=True)
    target  = interface.annoFormat(data=args.anno, uniqID=args.uniqID,
                                    mz=args.mzID, rt=args.rtID)
                                
    # Matching target file with library file
    logger.info("Identifying compounds")
    identified_df = identiyOnTarget(library=library, target=target, MZCut=0.005, RTCut=0.15)

    # Saving identified compounds to file
    identified_df.to_csv(args.output, sep="\t", index_label=args.uniqID)
    logger.info("Script Complete!")

if __name__=='__main__':
    #Get info
    args = getOptions()

    #Setting logger
    logger = logging.getLogger()
    sl.setLogger(logger)

    # Showing parameters
    logger.info("Importing data with following parameters:"
            "\n\tInput:  {0}"
            "\n\tLibrary:{1}".format(args.anno, args.library))

    # Runing script
    main(args)
