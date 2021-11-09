#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 14:46:34 2021

@author: zach 
@author: Alison updated with arguments for galaxy

"""
import argparse
import glob
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math

from upsetplot import UpSet


def getOptions():
    parser = argparse.ArgumentParser(description='Create plots and files for comparing pathway and in-Set meta-analyses')
    parser.add_argument("-i1", "--in1", dest="in1", action='store', required=True, help="Input summary file from pathway meta-analysis .")
    parser.add_argument("-p", "--inPath", dest="inPath", action='store', required=True, help="Name of metabolic pathway.")
    parser.add_argument("-m", "--inMut", dest="inMut", nargs='+', required=True, help="List of mutants (space separated) in the indicated metabolic pathway.")

    parser.add_argument("-i2", "--in2", dest="in2", action='store', required=True, help="Input directory containing summary files from in-Set meta-analysis .")

    parser.add_argument("-o", "--outD", dest="outD", action='store', required=True, help="Output directory.")
    parser.add_argument("--debug", action='store_true', default=False, help="Print debugging output")
    args=parser.parse_args()
    return args

def direction_determine(x):
    for i in x[1:]:
        if x[0] * i <=0:
            return 0
    return 1

def all_sig(x):
    for i in x:
        if i != 1:
            return 0
    return 1

def main():
    args = getOptions()

## set which pathway 
    group = args.inPath
    muts = args.inMut

    print(group)
    print(muts)


## summary file from running pathway
    combined = pd.read_csv(args.in1)
    #print("print pathway input")
    #print(combined)
    combined.columns = ["rowID"] + combined.columns[1:].tolist()
    combined = combined[["rowID", 'effect', 'p_value']]
    combined.columns = ['rowID', 'effect_meta_pathway', 'p_meta_pathway']

## summary file in-Set
    df_mut = combined
    mutpath = args.in2 + "/"
    print(mutpath)
    suffix='_summary.csv'
    for mut in muts:
        for file in os.listdir(mutpath):
            if mut in file and suffix in file:
                print(mut)
                f = os.path.join(mutpath, file)
                print(f)
                dat = pd.read_csv(f)
                dat.columns =  ["rowID"] + dat.columns[1:].tolist()
                dat = dat[['rowID', 'effect', 'p_value']]
                column_names = ['rowID'] + ['effect_' + mut] + ['p_' + mut]
                dat.columns = column_names
                #print("printing dat")
                #print(dat)
                df_mut = pd.merge(df_mut, dat, on = 'rowID', how = 'outer', indicator = "merge_check")
                #print(df_mut["merge_check"].value_counts())
                df_mut = df_mut[df_mut["merge_check"] != "right_only"].drop(columns=["merge_check"])
                #print("printing df_mut")
                #print(df_mut)

    df_mut.index = df_mut["rowID"]
    df_mut = df_mut.drop("rowID", axis = 1)

    es_cols = pd.Series(df_mut.columns).str.contains("effect|rowID").tolist()
    df_es = df_mut.loc[:,es_cols].copy()
    

    p_cols = pd.Series(df_mut.columns).str.contains("p_|rowID").tolist()
    df_p = df_mut.loc[:,p_cols]


    df_es["flag_same_direction"] = df_es.apply(direction_determine, axis = 1)
    #print("PRINT")
    #print(df_es) 
##### result 2: counts same direction
    counts_same_direct = sum(df_es["flag_same_direction"])
  
    sig_05_table = pd.DataFrame(np.where(df_p < 0.05, 1, 0))
    sig_05_table.columns = df_p.columns
    sig_05_table.index = df_p.index

    #sig_05_table.to_csv(args.outD + "/sig_table_" + group + ".csv")

##### result 1: count table for sig 0.05
    counts_sig_05 = pd.DataFrame(sig_05_table.apply(sum, axis = 0))
    counts_sig_05.columns = ['c_sig_0.05']

#####
    both_sig_same_05 = pd.Series(sig_05_table.loc[(df_es["flag_same_direction"] == 1) & (sig_05_table['p_meta_pathway'] == 1)].index)
    both_sig_same_05 = sig_05_table.apply(lambda x: len(df_p.loc[(df_es["flag_same_direction"] == 1) & (x ==1), x.name]))
    both_sig_same_05.name = 'both_sig_0.05_and_same_direct'

#####

####

    ## subset where pathway is significant in both dataframes
    sig_05_table_combine = sig_05_table[sig_05_table['p_meta_pathway'] == 1]
    df_es_sig_05_combine = df_es[sig_05_table["p_meta_pathway"] == 1]

    ## and is in the same direction
    df_es_sig_05_combine_sdirect = df_es_sig_05_combine[df_es_sig_05_combine["flag_same_direction"]==1]
    print(len(df_es_sig_05_combine_sdirect))

    ## and NOT in the same direction
    df_es_sig_05_combine_ndirect = df_es_sig_05_combine[df_es_sig_05_combine["flag_same_direction"]==0][100:200]

#######################
############ extract a effect size matrix for heatmap in R
    es_matrix = df_es[df_es['flag_same_direction'] == 1].copy()
    es_matrix['sig_counts'] = sig_05_table.loc[df_es['flag_same_direction'] == 1, ~sig_05_table.columns.str.contains("meta_pathway")].apply(sum, axis = 1)
    es_matrix['direction'] = np.where(es_matrix['effect_meta_pathway'] > 0, 1, 0)
    es_matrix = es_matrix.sort_values(by = ['sig_counts', 'direction'], ascending = False)
    sort_info = es_matrix[['sig_counts', 'direction']]

    es_matrix = es_matrix.merge(df_p["p_meta_pathway"], left_index = True, right_index = True, how = 'left')
    es_matrix.to_csv(args.outD + "/effect_size_matrix_" + group + ".csv")
    print(len(es_matrix))

####### scatter all same direction
    fig = plt.figure(figsize=(15,4), facecolor='white',edgecolor='black')
    ax = fig.add_subplot(111)
    colors = ['red', 'blue', 'green', 'orange', 'black', 'purple']
    labels = df_es_sig_05_combine.columns[:-1]
    #labels = pd.Series(labels).str.split("_").apply(lambda x: x[1])
    sizes = [3] + [1] * 5
    s = 0
    patches = []

    for i in df_es_sig_05_combine.columns[:-1]:
        patch = ax.plot([i for i in range(0, len(df_es_sig_05_combine_sdirect))],
        #patch = ax.plot([i * 2 for i in range(0, len(df_es_sig_05_combine_sdirect))], 
        df_es_sig_05_combine_sdirect.loc[:, i].tolist(),
       'o', markersize = sizes[s], color = colors[s])
        s +=1
        patches += patch

    ax.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1], color = 'black', linewidth = 0.5)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend([matplotlib.lines.Line2D([0],[0], color = x, marker = 'o', ls = '', markersize=3) for x in colors], 
              labels, loc='lower left', bbox_to_anchor=(1, 0.05), prop = dict(size = 8))
    plt.xlabel("individual features")
    plt.ylabel("effect size")
    fig.savefig(args.outD + "/scatterPlot_ES_sig_path_and_all_models_same_direct_" + group + '.pdf')
    plt.close(fig)

##### upset plot
    def plot_upset(df,title, fig, boxCols=None):
        """
        Plot an UpSet plot given a properly formatted dataframe (multi-indexed with boolean values)
        """
        upset = UpSet(df,subset_size='count',show_counts=True,sort_by='degree',sort_categories_by=None)
        if boxCols is not None:
            if type(boxCols) != list:
                boxCols = list(boxCols)
            for col in boxCols:
                upset.add_catplot(value=col, kind='box', elements = 4, showfliers=False, color=sns.color_palette('colorblind',15).as_hex()[boxCols.index(col)])
        upset.plot(fig = fig)
        plt.subplots_adjust(right=1.00001)
        plt.suptitle(title, fontsize=9)

    def split_pos_neg(sig_table, es_table):
        cols = sig_table.columns
        res = pd.DataFrame(index = sig_table.index)
        for col in cols:
            sig_list = sig_table[col]
            es_list = es_table[col]
            res[col + "_up"] = 0
            res[col + "_down"] = 0

            res[col+"_up"] = np.where(sig_list & es_list ,1,0) + res[col+'_up']
            res[col+"_down"] = np.where((sig_list == 1) & (es_list == 0), 1, 0) + res[col+'_down']
        return res


### 
    sig_upset = sig_05_table.iloc[:,1:]
    sig_upset = sig_upset[df_es['flag_same_direction'] == 0].copy()
    #sig_upset = sig_upset[~(df_es['flag_same_direction'] == 1)].copy()

    ## drop rows in sig_upset where all columns sum to 0
    sig_upset = sig_upset[~(sig_upset.apply(sum, axis = 1) == 0)]
### 
    df_es_direct = df_es.loc[sig_upset.index, df_es.columns[1:-1]]
    df_es_direct = pd.DataFrame(np.where(df_es_direct > 0, 1, 0))
    
    print(df_es_direct.shape) # 2, 4
    print(sig_upset.shape)    # 2, 6

    df_es_direct.index = sig_upset.index
    df_es_direct.columns = sig_upset.columns
    inputs = split_pos_neg(sig_upset, df_es_direct)

###
    cols = inputs.columns.to_list()
    inputs = inputs.reset_index()
    inputs[cols] = inputs[cols].astype(bool)
    inputs= inputs.set_index(cols)

###
    fig  = plt.figure(figsize=(12,8))
    plot_upset(inputs,"sig features where the ES is not consistent in direction between in-set and pathway models", fig)
    fig.savefig(args.outD + "/upset_num_sig_features_ES_inconsistent_inSet_vs_pathway_" + group + ".pdf", format='pdf')
    plt.close(fig)
        
if __name__ == '__main__':
    main()
