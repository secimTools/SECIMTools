#!/bin/bash
#     <test>
#        <param name="input"      value="ST000006_run_order_regression_flags.tsv"/>
#        <param name="flag1"      value="flag_feature_runOrder_pval_05" />
#        <param name="flag2"      value="flag_feature_runOrder_pval_01" />
#        <output name="output"    file="ST000006_compare_flags_output.tsv" />
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

compare_flags.py \
    -i  "$INPUT_DIR/ST000006_run_order_regression_flags.tsv" \
    -f1 flag_feature_runOrder_pval_05 \
    -f2 flag_feature_runOrder_pval_01 \
    -o  "$OUTPUT_DIR/ST000006_compare_flags_output.tsv"

echo "### Finished test: ${TEST} on $(date)"
