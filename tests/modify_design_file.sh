#!/bin/bash
#     <test>
#        <param name="input"  value="ST000006_data.tsv"/>
#        <param name="design" value="ST000006_design_group_name_underscore.tsv"/>
#        <param name="uniqID" value="Retention_Index" />
#        <param name="group"  value="White_wine_type_and_source_nosp" />
#        <param name="toDrop" value="Chardonnay_Napa_CA2003,Riesling_CA2004,SauvignonBlanc_LakeCounty_CA2001" />
#        <output name="out"   file="ST000006_modify_design_file_output.tsv" />
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

modify_design_file.py \
    -i  "$INPUT_DIR/ST000006_data.tsv" \
    -d  "$INPUT_DIR/ST000006_design_group_name_underscore.tsv" \
    -id Retention_Index \
    -g White_wine_type_and_source_nosp \
    -dp "Chardonnay_Napa_CA2003,Riesling_CA2004,SauvignonBlanc_LakeCounty_CA2001" \
    -o  "$OUTPUT_DIR/ST000006_modify_design_file_output.tsv"

echo "### Finished test: ${TEST} on $(date)"
