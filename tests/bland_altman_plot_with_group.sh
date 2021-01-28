#!/bin/bash
#     <test>
#        <param name="input"  value="ST000006_data.tsv"/>
#        <param name="design" value="ST000006_design.tsv"/>
#        <param name="uniqID" value="Retention_Index" />
#        <param name="group"  value="White_wine_type_and_source" />
#        <output name="ba_plots"           file="ST000006_bland_altman_plot_with_group_figure.pdf" compare="sim_size" delta="10000" />
#        <output name="outlier_dist_plots" file="ST000006_bland_altman_plot_with_group_flag_distribution.pdf" compare="sim_size" delta="10000" />
#        <output name="flag_sample"        file="ST000006_bland_altman_plot_with_group_flag_sample.tsv" />
#        <output name="flag_feature"       file="ST000006_bland_altman_plot_with_group_flag_feature.tsv" />
#     </test>

TEST="bland_altman_plot_with_group"
TESTDIR="testout/${TEST}"
rm -rf ${TESTDIR}
mkdir -p ${TESTDIR}
echo "### Starting test: ${TEST}"

INPUT_DIR="galaxy/test-data"
OUTPUT_DIR=$TESTDIR
if [[ $# -gt 0 ]]; then OUTPUT_DIR=$1 ; fi
mkdir -p "${OUTPUT_DIR}"

bland_altman_plot.py \
    -i  "$INPUT_DIR/ST000006_data.tsv" \
    -d  "$INPUT_DIR/ST000006_design.tsv" \
    -id Retention_Index \
    -g  White_wine_type_and_source \
    -f  "$OUTPUT_DIR/ST000006_bland_altman_plot_with_group_figure.pdf" \
    -fd "$OUTPUT_DIR/ST000006_bland_altman_plot_with_group_flag_distribution.pdf" \
    -fs "$OUTPUT_DIR/ST000006_bland_altman_plot_with_group_flag_sample.tsv" \
    -ff "$OUTPUT_DIR/ST000006_bland_altman_plot_with_group_flag_feature.tsv" \
    -pf "$OUTPUT_DIR/ST000006_bland_altman_plot_prop_feature.tsv" \
    -ps "$OUTPUT_DIR/ST000006_bland_altman_plot_prop_sample.tsv"

    date
echo "### Finished test: ${TEST}"
