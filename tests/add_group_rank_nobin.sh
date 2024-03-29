#!/bin/bash
#     <test>
#        <param name="wide"       value="ST000006_data.tsv"/>
#        <param name="design"      value="ST000006_design.tsv"/>
#        <param name="uniqID"      value="Retention_Index" />
#        <output name="out" file="ST000006_add_group_rank_output_wide.tsv" />
#     </test>

SCRIPT=$(basename "${BASH_SOURCE[0]}");
echo ${BASH_SOURCE[0]}
TEST="${SCRIPT%.*}"
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
    --out "$OUTPUT_DIR/ST000006_add_group_rank_noBin.tsv"

echo "### Finished test: ${TEST} on $(date)"
