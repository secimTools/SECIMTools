#!/bin/bash
#     <test>
#        <param name="input"       value="ST000006_data.tsv"/>
#        <param name="design"      value="ST000006_design.tsv"/>
#        <param name="uniqID"      value="Retention_Index" />
#        <param name="flags"       value="ST000006_kruskal_wallis_with_group_flags.tsv"/>
#        <param name="typeofdrop"  value="row"/>
#        <param name="flagUniqID"  value="Retention_Index" />
#        <param name="flagToDrop"  value="flag_significant_0p10_on_all_groups" />
#        <param name="reference"   value="0" />
#        <param name="conditional" value="0" />
#        <output name="outWide"    file="ST000006_remove_selected_features_samples_wide.tsv" />
#        <output name="outFlags"   file="ST000006_remove_selected_features_samples_flags.tsv" />
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

remove_selected_features_samples.py \
    -i  "$INPUT_DIR/ST000006_data.tsv" \
    -d  "$INPUT_DIR/ST000006_design_group_name_underscore.tsv" \
    -id Retention_Index \
    -f  "$INPUT_DIR/ST000006_kruskal_wallis_with_group_flags.tsv" \
    -fft row \
    -fid Retention_Index \
    -fd  flag_significant_0p10_on_all_groups \
    -val 0 \
    -con 0 \
    -ow "$OUTPUT_DIR/ST000006_remove_selected_features_samples_wide.tsv" \
    -of "$OUTPUT_DIR/ST000006_remove_selected_features_samples_flags.tsv"

echo "### Finished test: ${TEST} on $(date)"
