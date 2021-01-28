#!/bin/bash
#     <test>
#        <param name="input"       value="ST000006_data.tsv"/>
#        <param name="design"      value="ST000006_design.tsv"/>
#        <param name="uniqID"      value="Retention_Index" />
#        <param name="factor"      value="White_wine_type_and_source" />
#        <param name="factorTypes" value="C" />
#        <output name="results_table" file="ST000006_anova_fixed_with_group_summary.tsv" />
#        <output name="flags_table"   file="ST000006_anova_fixed_with_group_flags.tsv" />
#        <output name="qq_plots"      file="ST000006_anova_fixed_with_group_qq_plots.pdf" compare="sim_size" delta="10000" />
#        <output name="volcano_plots" file="ST000006_anova_fixed_with_group_volcano.pdf"  compare="sim_size" delta="10000" />
#     </test>

TEST="anova_fixed_with_group"
TESTDIR="testout/${TEST}"
rm -rf ${TESTDIR}
mkdir -p ${TESTDIR}
echo "### Starting test: ${TEST}"

INPUT_DIR="galaxy/test-data"
OUTPUT_DIR=$TESTDIR
if [[ $# -gt 0 ]]; then OUTPUT_DIR=$1 ; fi
mkdir -p "${OUTPUT_DIR}"

# Symlink into the conda bin directory
anova_fixed.py \
    -i  "$INPUT_DIR/ST000006_data.tsv" \
    -d  "$INPUT_DIR/ST000006_design.tsv" \
    -id Retention_Index \
    -f  White_wine_type_and_source \
    -t  C \
    -o  "$OUTPUT_DIR/ST000006_anova_fixed_with_group_summary.tsv" \
    -fl "$OUTPUT_DIR/ST000006_anova_fixed_with_group_flags.tsv" \
    -f1 "$OUTPUT_DIR/ST000006_anova_fixed_with_group_qq_plots.pdf" \
    -f2 "$OUTPUT_DIR/ST000006_anova_fixed_with_group_volcano.pdf"

date
echo "### Finished test: ${TEST}"
