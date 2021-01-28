#!/bin/bash
#     <test>
#        <param name="input"   value="ST000006_data.tsv"/>
#        <param name="design"  value="ST000006_design.tsv"/>
#        <param name="uniqID"  value="Retention_Index" />
#        <param name="corr"    value="pearson" />
#        <output name="output" file="ST000006_modulated_modularity_clustering_out.tsv" compare="sim_size" delta="10000"/>
#        <output name="figure" file="ST000006_modulated_modularity_clustering_figure.pdf" compare="sim_size" delta="10000" />
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

modulated_modularity_clustering.py \
    -i  "$INPUT_DIR/ST000006_data.tsv" \
    -d  "$INPUT_DIR/ST000006_design.tsv" \
    -id Retention_Index \
    -c pearson \
    -o "$OUTPUT_DIR/ST000006_modulated_modularity_clustering_out.tsv" \
    -f "$OUTPUT_DIR/ST000006_modulated_modularity_clustering_figure.pdf"

echo "### Finished test: ${TEST} on $(date)"
