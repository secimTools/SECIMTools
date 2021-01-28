#!/bin/bash
#     <test>
#        <param name="input"   value="ST000006_data.tsv"/>
#        <param name="design"  value="ST000006_design.tsv"/>
#        <param name="uniqID"  value="Retention_Index" />
#        <param name="group"   value="White_wine_type_and_source" />
#        <param name="penalty" value="0.5" />
#        <output name="plot"   file="ST000006_mahalanobis_distance_figure.pdf" compare="sim_size" delta="10000" />
#        <output name="out1"   file="ST000006_mahalanobis_distance_to_mean.tsv" />
#        <output name="out2"   file="ST000006_mahalanobis_distance_pairwise.tsv" />
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

mahalanobis_distance.py \
   -i  "$INPUT_DIR/ST000006_data.tsv" \
   -d  "$INPUT_DIR/ST000006_design.tsv" \
   -id Retention_Index \
   -g  White_wine_type_and_source \
   -pen 0.5 \
   -m  "$OUTPUT_DIR/ST000006_mahalanobis_distance_to_mean.tsv" \
   -f  "$OUTPUT_DIR/ST000006_mahalanobis_distance_figure.pdf" \
   -pw "$OUTPUT_DIR/ST000006_mahalanobis_distance_pairwise.tsv"

echo "### Finished test: ${TEST} on $(date)"
