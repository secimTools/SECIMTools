#!/bin/bash
#     <test>
#        <param name="wide"       value="the_1442_sbys_PH.tsv"/>
#        <param name="design"      value="design_gt_all_batches_w_pair.tsv"/>
#        <param name="uniqID"      value="rowID" />
#        <param name="ngroup" value=100 />
#        <output name="out" file="1442_add_group_rank_output_wide.tsv" />
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

add_group_rank.py \
    --wide "$INPUT_DIR/the_1442_sbys_PH.tsv" \
    --design  "$INPUT_DIR/design_gt_all_batches_w_pair.tsv" \
    --uniqID rowID \
    --out "$OUTPUT_DIR/1442_add_group_rank_output_wide.tsv"

echo "### Finished test: ${TEST} on $(date)"
