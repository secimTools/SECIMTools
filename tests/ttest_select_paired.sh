#!/bin/bash
#     <test>
#        <param name="input"     value="ST000006_data.tsv"/>
#        <param name="design"    value="ST000006_design.tsv"/>
#        <param name="uniqueID"  value="Retention_Index" />
#        <param name="group"     value="White_wine_type_and_source" />
#        <param name="pairing"   value="unpaired" />
#        <output name="summaries" file="ST000006_ttest_select_unpaired_summary.tsv" />
#        <output name="flags"     file="ST000006_ttest_select_unpaired_flags.tsv" />
#        <output name="volcano"   file="ST000006_ttest_select_unpaired_volcano.pdf" compare="sim_size" delta="10000"/>
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

ttest.py \
    -i  "$INPUT_DIR/data_4_ttest_pair.tsv" \
    -d  "$INPUT_DIR/design_4_ttest_pair.tsv" \
    -id "Retention_Index" \
    -g  "group" \
    -p  "paired" \
    -o  "fake_pairID" \
    -s  "$OUTPUT_DIR/ST000006_ttest_select_paired_summary.tsv" \
    -f  "$OUTPUT_DIR/ST000006_ttest_select_paired_flags.tsv" \
    -v  "$OUTPUT_DIR/ST000006_ttest_select_paired_volcano.pdf"

echo "### Finished test: ${TEST} on $(date)"
