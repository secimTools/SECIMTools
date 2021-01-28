#!/bin/bash
#     <test>
#        <param name="input"   value="ST000006_data.tsv"/>
#        <param name="design"  value="ST000006_design.tsv"/>
#        <param name="uniqID"  value="Retention_Index" />
#        <param name="group"   value="White_wine_type_and_source" />
#        <param name="cutoff"  value="3000" />
#        <output name="output" file="ST000006_threshold_based_flags_output.tsv" />
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

threshold_based_flags.py \
    -i  "$INPUT_DIR/ST000006_data.tsv" \
    -d  "$INPUT_DIR/ST000006_design.tsv" \
    -id "Retention_Index" \
    -g  "White_wine_type_and_source" \
    -c  "3000" \
    -o  "$OUTPUT_DIR/ST000006_threshold_based_flags_output.tsv"

echo "### Finished test: ${TEST} on $(date)"
