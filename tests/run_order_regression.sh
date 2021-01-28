#!/bin/bash
#     <test>
#        <param name="input"  value="ST000006_data.tsv"/>
#        <param name="design" value="ST000006_design.tsv"/>
#        <param name="uniqID" value="Retention_Index" />
#        <param name="group"  value="White_wine_type_and_source" />
#        <param name="order"  value="run_Order_fake_variable" />
#        <output name="figure"        file="ST000006_run_order_regression_figure.pdf" compare="sim_size" delta="10000" />
#        <output name="order_summary" file="ST000006_run_order_regression_table.tsv" />
#        <output name="flags"         file="ST000006_run_order_regression_flags.tsv" />
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

run_order_regression.py \
    -i  "$INPUT_DIR/ST000006_data.tsv" \
    -d  "$INPUT_DIR/ST000006_design.tsv" \
    -id Retention_Index \
    -g  White_wine_type_and_source \
    -o  run_Order_fake_variable \
    -f  "$OUTPUT_DIR/ST000006_run_order_regression_figure.pdf" \
    -t  "$OUTPUT_DIR/ST000006_run_order_regression_table.tsv" \
    -fl "$OUTPUT_DIR/ST000006_run_order_regression_flags.tsv"

echo "### Finished test: ${TEST} on $(date)"
