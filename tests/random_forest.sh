#!/bin/bash
#     <test>
#        <param name="input"  value="ST000006_data.tsv"/>
#        <param name="design" value="ST000006_design.tsv"/>
#        <param name="uniqID" value="Retention_Index" />
#        <param name="group"  value="White_wine_type_and_source" />
#        <output name="outfile1" file="ST000006_random_forest_out.tsv" />
#        <output name="outfile2" file="ST000006_random_forest_out2.tsv" />
#        <output name="figure"   file="ST000006_random_forest_figure.pdf" compare="sim_size" delta="10000"/>
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

random_forest.py \
    -i  "$INPUT_DIR/ST000006_data.tsv" \
    -d  "$INPUT_DIR/ST000006_design.tsv" \
    -id Retention_Index \
    -g White_wine_type_and_source \
    -o  "$OUTPUT_DIR/ST000006_random_forest_out.tsv" \
    -o2 "$OUTPUT_DIR/ST000006_random_forest_out2.tsv" \
    -f  "$OUTPUT_DIR/ST000006_random_forest_figure.pdf"

echo "### Finished test: ${TEST} on $(date)"
