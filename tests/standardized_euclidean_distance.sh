#!/bin/bash
#      <test>
#        <param name="input"  value="ST000006_data.tsv"/>
#        <param name="design" value="ST000006_design.tsv"/>
#        <param name="uniqID" value="Retention_Index" />
#        <param name="group"  value="White_wine_type_and_source" />
#        <output name="plot"  value="ST000006_standardized_euclidean_distance.pdf" compare="sim_size" delta="50000" />
#        <output name="out1"  file="ST000006_standardized_euclidean_distance_to_mean.tsv" />
#        <output name="out2"  file="ST000006_standardized_euclidean_distance_pairwise.tsv" />
#     </test>

SCRIPT=$(basename "${BASH_SOURCE[0]}");
TEST="${SCRIPT%.*}"
TESTDIR="testout/${TEST}"
INPUT_DIR="galaxy/test-data"
OUTPUT_DIR=$TESTDIR
rm -rf "${TESTDIR}"
mkdir -p "${TESTDIR}"
echo "### Starting test: ${TEST}"
if [[ $# -gt 0 ]]; then OUTPUT_DIR=$1 ; fi
mkdir -p "${OUTPUT_DIR}"

standardized_euclidean_distance.py \
        -i  "$INPUT_DIR/ST000006_data.tsv" \
        -d  "$INPUT_DIR/ST000006_design.tsv" \
        -id Retention_Index \
        -g  White_wine_type_and_source \
        -m  "$OUTPUT_DIR/ST000006_standardized_euclidean_distance_to_mean.tsv" \
        -f  "$OUTPUT_DIR/ST000006_standardized_euclidean_distance_figure.pdf" \
        -pw "$OUTPUT_DIR/ST000006_standardized_euclidean_distance_pairwise.tsv"

echo "### Finished test: ${TEST} on $(date)"
