#!/bin/bash
SCRIPT=$(basename "${BASH_SOURCE[0]}"); NAME="${SCRIPT%.*}"
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
INPUT_DIR="${SCRIPT_DIR}/../test-data"
OUTPUT_DIR="$(pwd)/output/${NAME}_${TIMESTAMP}"
if [[ $# -gt 0 ]]; then OUTPUT_DIR=$1 ; fi
mkdir -p "${OUTPUT_DIR}"

compare_flags.py \
    -i  "$INPUT_DIR/ST000006_run_order_regression_flags.tsv" \
    -f1 flag_feature_runOrder_pval_05 \
    -f2 flag_feature_runOrder_pval_01 \
    -o  "$OUTPUT_DIR/ST000006_compare_flags_output.tsv"

: '

The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="input"      value="ST000006_run_order_regression_flags.tsv"/>
        <param name="flag1"      value="flag_feature_runOrder_pval_05" />
        <param name="flag2"      value="flag_feature_runOrder_pval_01" />
        <output name="output"    file="ST000006_compare_flags_output.tsv" />
     </test>
    </tests>

'
