#!/bin/bash
#     <test>
#        <param name="input"  value="ST000006_anova_fixed_with_group_summary.tsv"/>
#        <param name="uniqID" value="Retention_Index" />
#        <param name="pval"   value="prob_greater_than_t_for_diff_Chardonnay, Carneros, CA 2003 (CH01)-Chardonnay, Carneros, CA 2003 (CH02)" />
#        <param name="alpha"  value="0.05" />
#        <output name="outadjusted" file="ST000006_multiple_testing_adjustment_outadjusted.tsv" />
#        <output name="flags"       file="ST000006_multiple_testing_adjustment_flags.tsv" />
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

multiple_testing_adjustment.py \
    -i  "$INPUT_DIR/ST000006_anova_fixed_with_group_summary.tsv" \
    -id Retention_Index \
    -pv "prob_greater_than_t_for_diff_Chardonnay, Carneros, CA 2003 (CH01)-Chardonnay, Carneros, CA 2003 (CH02)" \
    -a  0.05 \
    -on "$OUTPUT_DIR/ST000006_multiple_testing_adjustment_outadjusted.tsv" \
    -fl "$OUTPUT_DIR/ST000006_multiple_testing_adjustment_flags.tsv"

echo "### Finished test: ${TEST} on $(date)"
