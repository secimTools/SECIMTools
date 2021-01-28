#!/bin/bash
#     <test>
#        <param name="input"  value="ST000006_data.tsv"/>
#        <param name="design" value="ST000006_design.tsv"/>
#        <param name="uniqID" value="Retention_Index" />
#        <param name="group"  value="White_wine_type_and_source" />
#        <param name="alpha"  value="0.5" />
#        <output name="coefficients" file="ST000006_lasso_enet_var_select_coefficients.tsv" />
#        <output name="flags"        file="ST000006_lasso_enet_var_select_flags.tsv" />
#        <output name="plots"        file="ST000006_lasso_enet_var_select_plots.pdf" compare="sim_size" delta="10000" />
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

lasso_enet_var_select.py \
        -i  "$INPUT_DIR/ST000006_data.tsv" \
        -d  "$INPUT_DIR/ST000006_design.tsv" \
        -id Retention_Index \
        -g White_wine_type_and_source \
        -a 0.5 \
        -c  "$OUTPUT_DIR/ST000006_lasso_enet_var_select_coefficients.tsv" \
        -f  "$OUTPUT_DIR/ST000006_lasso_enet_var_select_flags.tsv" \
        -p  "$OUTPUT_DIR/ST000006_lasso_enet_var_select_plots.pdf"

echo "### Finished test: ${TEST} on $(date)"
