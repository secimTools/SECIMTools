#!/bin/bash
#     <test>
#        <param name="input"  value="ST000006_data.tsv"/>
#        <param name="design" value="ST000006_design.tsv"/>
#        <param name="uniqID" value="Retention_Index" />
#        <param name="group"  value="White_wine_type_and_source" />
#        <output name="loadings" file="ST000006_principal_component_analysis_load_out.tsv" />
#        <output name="scores"   file="ST000006_principal_component_analysis_score_out.tsv" />
#        <output name="summary"  file="ST000006_principal_component_analysis_summary_out.tsv" />
#        <output name="figures"  file="ST000006_principal_component_analysis_figure.pdf" compare="sim_size" delta="10000"/>
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

principal_component_analysis.py \
    -i  "$INPUT_DIR/ST000006_data.tsv" \
    -d  "$INPUT_DIR/ST000006_design.tsv" \
    -id Retention_Index \
    -g White_wine_type_and_source \
    -lo "$OUTPUT_DIR/ST000006_principal_component_analysis_load_out.tsv" \
    -so "$OUTPUT_DIR/ST000006_principal_component_analysis_score_out.tsv" \
    -su "$OUTPUT_DIR/ST000006_principal_component_analysis_summary_out.tsv" \
    -f  "$OUTPUT_DIR/ST000006_principal_component_analysis_figure.pdf"

echo "### Finished test: ${TEST} on $(date)"
