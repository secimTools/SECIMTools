#!/bin/bash
#      <test>
#        <param name="input"  value="ST000006_data.tsv"/>
#        <param name="design" value="ST000006_design.tsv"/>
#        <param name="uniqID" value="Retention_Index" />
#        <param name="method" value="mean" />
#        <output name="out"   file="ST000006_data_normalization_and_rescaling_mean_output.tsv" />
#      </test>

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

data_normalization_and_rescaling.py \
    -i  "$INPUT_DIR/ST000006_data.tsv" \
    -d  "$INPUT_DIR/ST000006_design.tsv" \
    -id Retention_Index \
    -m  mean \
    -o "$OUTPUT_DIR/ST000006_data_normalization_and_rescaling_mean_output.tsv"

echo "### Finished test: ${TEST} on $(date)"
