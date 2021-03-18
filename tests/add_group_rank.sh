#!/bin/bash
#     <test>
# MISSING
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
#FIXME
#    -i  "$INPUT_DIR/ST000006_data.tsv" \
#    -d  "$INPUT_DIR/ST000006_design.tsv" \
#    -id Retention_Index \
#    -f  White_wine_type_and_source \
#    -t  C \
#    -o  "$OUTPUT_DIR/ST000006_anova_fixed_with_group_summary.tsv" \
#    -fl "$OUTPUT_DIR/ST000006_anova_fixed_with_group_flags.tsv" \
#    -f1 "$OUTPUT_DIR/ST000006_anova_fixed_with_group_qq_plots.pdf" \
#    -f2 "$OUTPUT_DIR/ST000006_anova_fixed_with_group_volcano.pdf"

echo "### Finished test: ${TEST} on $(date)"
