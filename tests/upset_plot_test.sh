#!/bin/bash
## use SECIMTools conda env
export PATH=/ufgi-serrata/data/cjohnson/cj_conda_envs/__secimtools@21.2.21/bin:$PATH

PROJ=/ufgi-serrata/data/cjohnson/SECIMTools
SCRIPT=/ufgi-serrata/data/cjohnson/SECIMTools/src/script
OUTDIR=/ufgi-serrata/data/cjohnson/Upset_testing
#INPUT=~/mclab/SHARE/McIntyre_Lab/Transcript_distances/rerun_analysis/MaizeWang2020/test_trand_chr10/TranD_2GTF_B73_R_vs_B73_END_chr10


python $SCRIPT/upset_plot.py \
    --wide ~/mclab/SHARE/McIntyre_Lab/Transcript_distances/rerun_analysis/MaizeWang2020/test_trand_chr10/TranD_2GTF_B73_R_vs_B73_END_chr10/pairwise_transcript_distance.csv \
    --uniqID rowID \
    --title upset_test \
    --design $PROJ/galaxy/test-data/upset_plot_design.csv \
    --outD $OUTDIR

