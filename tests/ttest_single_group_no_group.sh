#!/bin/bash
#      <test>
#        <param name="input"      value="ST000006_data.tsv"/>
#        <param name="design"     value="ST000006_design.tsv"/>
#        <param name="uniqueID"   value="Retention_Index" />
#        <param name="mu"         value="0" />
#        <output name="summaries" file="ST000006_ttest_single_group_no_group_summary.tsv" />
#        <output name="flags"     file="ST000006_ttest_single_group_no_group_flags.tsv" />
#        <output name="volcano"   file="ST000006_ttest_single_group_no_group_volcano.pdf" compare="sim_size" delta="10000" />
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

ttest_single_group.py \
    -i  "$INPUT_DIR/ST000006_data.tsv" \
    -d  "$INPUT_DIR/ST000006_design.tsv" \
    -id Retention_Index \
    -mu 0 \
    -s  "$OUTPUT_DIR/ST000006_ttest_single_group_no_group_summary.tsv" \
    -f  "$OUTPUT_DIR/ST000006_ttest_single_group_no_group_flags.tsv" \
    -v  "$OUTPUT_DIR/ST000006_ttest_single_group_no_group_volcano.pdf"

echo "### Finished test: ${TEST} on $(date)"
