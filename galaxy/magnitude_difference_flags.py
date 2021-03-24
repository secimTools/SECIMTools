#!/usr/bin/env python
################################################################################
# DATE: 2016/May/06, rev: 2016/July/11
#
# SCRIPT: magnitude_difference_flags.py
#
# VERSION: 2.0
#
# AUTHOR: Miguel A Ibarra (miguelib@ufl.edu)
#
# DESCRIPTION: This script takes a a wide format file and counts digits in 
# decimal numbers.The output is an html file containing graphs and data
#
################################################################################
# Import built-in libraries
import os
import logging
import zipfile
import argparse
from io import StringIO

# Import add-on libraries
import matplotlib
import numpy as np
matplotlib.use('Agg')
from lxml import etree
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from matplotlib.backends.backend_pdf import PdfPages

# Import local data libraries
from secimtools.dataManager import logger as sl
from secimtools.dataManager.flags import Flags
from secimtools.dataManager.interface import wideToDesign

# Import local plotting libraries
from secimtools.visualManager import module_hist as hist
from secimtools.visualManager.manager_color import colorHandler
from secimtools.visualManager.manager_figure import figureHandler

def getOptions(myopts=None):
    """ Function to pull in arguments """
    description = """ Count the digits in data to determine possible outliers
                      or discrepancies"""
    parser = argparse.ArgumentParser(description=description)
    # Standard input
    standard = parser.add_argument_group(description="Standar input")
    standard.add_argument('-i',"--input", dest="input", action='store', 
                        required=True, help="Input dataset in wide format.")
    standard.add_argument('-d',"--design", dest="design", action='store', 
                        required=True, help="Design file.")
    standard.add_argument('-id',"--ID", dest="uniqID", action='store', 
                        required=True, help="Name of the column with uniq IDs.")
    standard.add_argument('-g',"--group", dest="group", action='store', 
                         required=False, default=False, help="Add the option to "\
                        "separate sample IDs by treatement name. ")
    # Tool input
    tool = parser.add_argument_group(description="Optional input")
    tool.add_argument('-nz',"--noZero", dest="zero", action='store_true', 
                        required=False, help="Flag to ignore zeros.")
    tool.add_argument('-bug',"--debug", dest="debug", action='store_true', 
                        required=False, help="Add debugging log output.")
    tool.add_argument('-ht',"--html", dest="html", action='store', 
                        required=False, default=None,  help="Path for html"\
                        " output file (this option is just for galaxy")
    tool.add_argument('-htp',"--htmlPath", dest="htmlPath", action='store', 
                        required=False, default=None,  help="Path for html "\
                        "output file (this option is just for galaxy")
    # Tool output
    output = parser.add_argument_group(description="Output options")
    output.add_argument("-f","--figure",dest="figure",action="store",
                        required=True,help="Output path for plot file")
    output.add_argument("-fl","--flags",dest="flags",action="store",
                        required=True,help="Output path for flag file")
    output.add_argument("-c","--counts",dest="counts",action="store",
                        required=True,help="Output name for counts files"\
                        "The extension is not requiered its going to be added"\
                        "automatically for each file.")
    if myopts:
        args = parser.parse_args(myopts)
    else:
        args = parser.parse_args()

    return(args)

def splitDigits(x):
    """ 
    Function to split digits by decimal

        :Arguments:
            :type x: int
            :param x: Number to count digits form.

        :Returns:
            :rtype x: int
            :returns x: count of the given number.

    """
    if x == 0:
        return np.nan
    else:
        # Bug fixer of scientific notation (Very large and very small numbers)
        x = str('%f' % x)

        # Split x at the decimal point and then take the length of the string
        # Before the decimal point then return
        return len(x.split('.')[0])

def countDigits(wide):
    """
    This function counts digits on a given file.

        :Arguments:
            :type wide: pandas.DataFrame.
            :param wide: Input data to count digits.

        :Returns:
            :rtype count: pandas.DataFrame
            :returns count: DataFrama with the counted digits and min, max and 
                            diff among rows.

    """
    # Count the number of digits before decimal and get basic distribution info
    count = wide.applymap(lambda x: splitDigits(x))

    # Calculate min, max number of digits on the row and the difference
    count["min"] = count.apply(np.min, axis=1)
    count["max"] = count.apply(np.max, axis=1)
    count["diff"] = count["max"] - count["min"]

    # Return counts
    return count

def plotCDhistogram(count,pdf,group):
    """
    This function counts digits on a given file.

        :Arguments:
            :type count: pandas.DataFrame.
            :param count: DataFrama with the counted digits and min, max and 
                            diff among rows.

            :type pdf: matplotlib.backends.backend_pdf.PdfPages.
            :param pdf: PDF object to plot figures in.

            :type group: str.
            :param group: Name of the group to plot.
    """
    #Creating title
    title="Distribution of difference between \n(min and max) for {0} compounds".\
            format(group)
    if count['diff'].any():
        
        #Opening figure handler
        fh = figureHandler(proj='2d')

        #Plot histogram
        hist.quickHist(ax=fh.ax[0],dat=count['diff'])

        #Giving format to the axis
        fh.formatAxis(xTitle='Difference in Number of Digits (max - min)',
            yTitle='Number of Features',figTitle=title, ylim="ignore")

        # Explort figure
        fh.addToPdf(pdf,dpi=600)

    else:
        logger.warn("There were no differences in digit counts for {0}, no plot will be generated".format(group))

def createHTML():
    #Create html object
    html = etree.Element("html")
    head = etree.SubElement(html, "head")
    title = etree.SubElement(head, "title")
    title.text = "Count Digits Results List"
    body = etree.SubElement(html, "body")
    div = etree.SubElement(body, "div",style="background-color:black; "\
                "color:white; text-align:center; margin-bottom:5% padding:4px;")
    h1 = etree.SubElement(div,"h1")
    ul = etree.SubElement(body,"ul",style="text-align:left; margin-left:5%;")
    h1.text="Output"
    
    #Return htmml
    return html

def save2html(data, filename, filePath, html=None):
    #Add data to html
    if html is not None:
        li = etree.SubElement(html[1][1],"li", style="margin-bottom:1.5%;")
        a= etree.SubElement(li,"a",href=filename)
        a.text=os.path.split(filename)[1]

    #Save data
    data.to_csv(filePath,sep="\t",na_rep=0)

    #Return html
    return html

def countDigitsByGroup(dat, args, folderDir, pdf, html=None):
    """ 
    If the group option is selected this function is called to split by groups.

    The function calls the countDigits function in a loop that iterates through
    the groups

        :Arguments:
            :type dat: wideToDesign
            :param dat: input data 

            :type args: argparse.ArgumentParser.
            :param args: Command line arguments.

            :type pdf: matplotlib.backends.backend_pdf.PdfPages.
            :param pdf: PDF object to plot figures in.

            :type countzip: zipfile.ZipFile.
            :param countzip: Zip container.
    """
    # Split Design file by group
    if dat.group:
        for name, group in dat.design.groupby(dat.group):
            # Setting count path
            countPath = folderDir+"_{0}.tsv".format(name)

            # Setting count name
            countName = args.counts+"_{0}.tsv".format(name)

            # Filter the wide file into a new dataframe
            currentFrame = dat.wide[group.index]

            # Counting digits per group
            count  = countDigits(currentFrame)

            # Plotting CD histograms
            plotCDhistogram(count,pdf,name)

            # Save countName, save it to html if exist
            save2html(html=html, data=count, filename=countName, filePath=countPath)
            
def saveFlags(count):
    """ 
    Function to create and export flags for the counts.

        :Arguments:
            :type count: pandas.DataFrame.
            :param count: DataFrama with the counted digits and min, max and 
                            diff among rows.
    """

    # Create flag object
    flag = Flags(index=count.index)

    # If the difference is greater than 1 a flag is set for dat row/met.
    flag.addColumn(column="flag_feature_count_digits",mask=count["diff"] >= 2)

    #Save flags
    flag.df_flags.to_csv(os.path.abspath(args.flags),sep="\t")

def main(args):
    #parsing data with interface
    dat = wideToDesign(wide=args.input, design=args.design, uniqID=args.uniqID, 
                        group=args.group, logger=logger)

    # Removing groups with just one elemen from dat
    dat.removeSingle()

    # Create folder for counts if html found
    if args.html is not None:
        logger.info(u"Using html output file")
        folderDir = args.htmlPath
        try:
            os.makedirs(folderDir)
#KAS: changing raising exception error. Gives syntax error
#        except Exception, e:
#        raise Exception (e)
        except Exception as e:
            logger.error("Error. {}".format(e))

        # Initiation zip files
        html      = createHTML()
        folderDir =folderDir+"/"+args.counts
    else:
        folderDir=args.counts
        html     =args.html

    # Use group separation or not depending on user input
    with PdfPages(os.path.abspath(args.figure)) as pdf:
        if args.group:
            # Count Digits per group
            logger.info(u"Counting digits per group")
            countDigitsByGroup(dat, args, folderDir, pdf, html=html)

        # Count digits for all elements
        count = countDigits(wide=dat.wide)

        # Plotting for all elements
        plotCDhistogram(count=count, pdf=pdf, group="all")

        # Calculate and save flags for all elements
        logger.info(u"Calculating flags")
        saveFlags(count)

    # Add count of all elements to html if not html save directly
    html = save2html(html=html,data=count,filename=args.counts+"_all.tsv",filePath=folderDir+"_all.tsv")

    #Save to html
    if args.html:
        with open(args.html,"w") as  htmlOut:
            print(etree.tostring(html,pretty_print=True), file=htmlOut)
            # print >> htmlOut,etree.tostring(html,pretty_print=True)

    # Finishing script    
    logger.info(u"Count Digits Complete!")

if __name__ == '__main__':
    # Command line options
    args = getOptions()

    # Setting logger
    logger = logging.getLogger()
    if args.debug:
        sl.setLogger(logger, logLevel='debug')
    else:
        sl.setLogger(logger)

    # Starting script with the following parameters
    logger.info(u"Importing data with following parameters: "\
                "\n\tWide: {0}"\
                "\n\tDesign: {1}"\
                "\n\tUnique ID: {2}"\
                "\n\tGroup: {3}"\
                "\n\tHtml: {4}".\
    format(args.input,args.design, args.uniqID, args.group, args.html))

    # Main
    main(args)
