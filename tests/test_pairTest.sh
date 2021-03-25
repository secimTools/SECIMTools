#!/bin/bash

INPUT_DIR=/ufgi-serrata/data/zihaoliu/galaxy/tools/SECIMTools/galaxy/test-data
OUTPUT_DIR=/ufgi-serrata/data/zihaoliu/galaxy/tools/SECIMTools/galaxy/test-data/zihao

ttest.py \
    -i  "$INPUT_DIR/data_4_ttest_pair.tsv" \
    -d  "$INPUT_DIR/design_4_ttest_pair.tsv" \
    -id "Retention_Index" \
    -g  "group" \
    -p  "paired" \
    -o  "fake_pairID" \
    -s  "$OUTPUT_DIR/ST000006_ttest_select_paired_summary.tsv" \
    -f  "$OUTPUT_DIR/ST000006_ttest_select_paired_flags.tsv" \
    -v  "$OUTPUT_DIR/ST000006_ttest_select_paired_volcano.pdf"

echo "### Finished test: ${TEST} on $(date)"
