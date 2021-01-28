#!/bin/bash
#     <test>
#        <param name="input"      value="ST000006_run_order_regression_flags.tsv,ST000006_lasso_enet_var_select_flags.tsv"/>
#        <param name="flagUniqID" value="Retention_Index" />
#        <param name="filename"   value="ST000006_run_order_regression_flags ST000006_lasso_enet_var_select_flags" />
#        <output name="output"    file="ST000006_merge_flags_output.tsv" />
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

merge_flags.py \
    -i  "$INPUT_DIR/ST000006_run_order_regression_flags.tsv,$INPUT_DIR/ST000006_lasso_enet_var_select_flags.tsv" \
    -f  "ST000006_run_order_regression_flags_tsv" "ST000006_lasso_enet_var_select_flags_tsv" \
    -fid Retention_Index \
    -o  "$OUTPUT_DIR/ST000006_merge_flags_output.tsv" \

echo "### Finished test: ${TEST} on $(date)"
