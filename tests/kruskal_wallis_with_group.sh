#!/bin/bash
#     <test>
#        <param name="input"    value="ST000006_data.tsv"/>
#        <param name="design"   value="ST000006_design.tsv"/>
#        <param name="uniqueID" value="Retention_Index" />
#        <param name="group"    value="White_wine_type_and_source" />
#        <output name="summaries" file="ST000006_kruskal_wallis_with_group_summary.tsv" />
#        <output name="flags"     file="ST000006_kruskal_wallis_with_group_flags.tsv" />
#        <output name="volcano"   file="ST000006_kruskal_wallis_with_group_volcano.pdf" compare="sim_size" delta="10000" />
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

kruskal_wallis.py \
    -i  "$INPUT_DIR/ST000006_data.tsv" \
    -d  "$INPUT_DIR/ST000006_design.tsv" \
    -id Retention_Index \
    -g White_wine_type_and_source \
    -s "$OUTPUT_DIR/ST000006_kruskal_wallis_with_group_summary.tsv" \
    -f "$OUTPUT_DIR/ST000006_kruskal_wallis_with_group_flags.tsv" \
    -v "$OUTPUT_DIR/ST000006_kruskal_wallis_with_group_volcano.pdf"

echo "### Finished test: ${TEST} on $(date)"
