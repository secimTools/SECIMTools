#!/bin/bash
#     <test>
#        <param name="flags"      value="ST000006_lasso_enet_var_select_flags.tsv"/>
#        <param name="uniqID"     value="Retention_Index" />
#        <output name="output"    file="ST000006_summarize_flags_outSummary.tsv" />
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

summarize_flags.py \
    -f  "$INPUT_DIR/ST000006_lasso_enet_var_select_flags.tsv" \
    -id "Retention_Index" \
    -os "$OUTPUT_DIR/ST000006_summarize_flags_outSummary.tsv"

echo "### Finished test: ${TEST} on $(date)"
