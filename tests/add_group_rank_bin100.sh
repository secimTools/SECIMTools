#!/bin/bash
#     <test>
#        <param name="wide"       value="ST000006_data.tsv"/>
#        <param name="design"      value="ST000006_design.tsv"/>
#        <param name="uniqID"      value="Retention_Index" />
#        <param name="ngroup" value=100 />
#        <output name="out" file="ST000006_add_group_rank_output_wide.tsv" />
#     </test>

SCRIPT=$(basename "${BASH_SOURCE[0]}");
TEST="${SCRIPT%.*}"
echo "Echoing ${BASH_SOURCE[0]}"
TESTDIR="testout/${TEST}"
INPUT_DIR="galaxy/test-data"
OUTPUT_DIR=$TESTDIR
rm -rf "${TESTDIR}"
mkdir -p "${TESTDIR}"
echo "### Starting test: ${TEST}"
if [[ $# -gt 0 ]]; then OUTPUT_DIR=$1 ; fi
mkdir -p "${OUTPUT_DIR}"

add_group_rank.py \
    --wide "$INPUT_DIR/ST000006_data.tsv" \
    --design  "$INPUT_DIR/ST000006_design.tsv" \
    --uniqID Retention_Index \
    --ngroup 100 \
    --out "$OUTPUT_DIR/ST000006_add_group_rank_bin100.tsv"

echo "### Finished test: ${TEST} on $(date)"
