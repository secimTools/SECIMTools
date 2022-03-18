#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 14:24:05 2021

@author: carter.johnson
"""

import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from upsetplot import UpSet


def getOptions():
    parser = argparse.ArgumentParser(description='Creates upset plot and outputs a pdf')
    parser.add_argument('-w', '--wide', dest='wide', required=True, help="The input file in wide format. Must be tsv")
    parser.add_argument('-d', '--design', dest='design', required=True, help="A design file that assigns columns in wide datsaset to be plotted and specifies the labels for the plot.")
    parser.add_argument('-u', '--uniqID', dest='uniqID', default='rowID', help="Specify the unique row ID column in the wide input dataset")
    parser.add_argument('-t', '--title', dest ='title', required =True, help="Specify the title to use in plot pdf")
    parser.add_argument("-o", "--outD", dest="outD", action='store', required=True, help="Specify pdf output plot")
    parser.add_argument("-m", "--minSS", dest="minSS", default='0', help="Set minimum subset size - use to truncate graph (e.g. omit 1's)")
    args=parser.parse_args()
    return args

def plot_upset(df, title, fig, boxCols=None):
        """
        Plot an UpSet plot given a properly formatted dataframe (multi-indexed with boolean values)
        """
        args = getOptions()
        minSS = args.minSS
        
        upset = UpSet(df,subset_size='count',show_counts=True,sort_by='cardinality',sort_categories_by=None, min_subset_size=int(minSS))
#        upset = UpSet(df,subset_size='count',show_counts=True,sort_by='cardinality',sort_categories_by=None, min_subset_size=2)
#        upset = UpSet(df,subset_size='count',show_counts=True,sort_by='cardinality',sort_categories_by=None)
        
        if boxCols is not None:
            if type(boxCols) != list:
                boxCols = list(boxCols)
            for col in boxCols:
                upset.add_catplot(value=col, kind='box', elements = 4, showfliers=False, color=sns.color_palette('colorblind',15).as_hex()[boxCols.index(col)])
        upset.plot(fig = fig)
        plt.subplots_adjust(right=1.00001)
        plt.suptitle(title, fontsize=9)
        upset.plot(fig = fig)
        plt.subplots_adjust(right=1.00001)
        plt.suptitle(title, fontsize=9)
        
        
        
def main():
    args = getOptions()
    uniqID = args.uniqID
    title = args.title
    design = pd.read_csv(args.design, sep='\t')


    #Extract list of desired columns and desired names to a list from design file
    data_cols = design.iloc[:,0].tolist()
    colNameList = design.iloc[:,1].tolist()
    
    df = pd.read_csv(args.wide, sep='\t') #Input must be tsv file
        
    #Eliminates columns of unwanted data
    df = df[data_cols]
    #Rename Columns
    df = df.set_axis(colNameList, axis='columns')
    #Reset index to create a multi-indexed boolean series
    df = df.reset_index()
    df[colNameList] = df[colNameList].astype(bool)
    df = df.set_index(colNameList)
    
    fig  = plt.figure(figsize=(12,8))
    plot_upset(df, title, fig)
    fig.savefig(args.outD, format='pdf')
#    fig.savefig(args.outD + ".svg", format='svg')
    plt.close(fig)
    
if __name__ == '__main__':
    main()
