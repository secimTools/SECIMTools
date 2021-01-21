#!/bin/bash
SCRIPT=$(basename "${BASH_SOURCE[0]}"); NAME="${SCRIPT%.*}"
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
INPUT_DIR="${SCRIPT_DIR}/../test-data"
OUTPUT_DIR="$(pwd)/output/${NAME}_${TIMESTAMP}"
if [[ $# -gt 0 ]]; then OUTPUT_DIR=$1 ; fi
mkdir -p "${OUTPUT_DIR}"

kruskal_wallis.py \
    -i  "$INPUT_DIR/ST000006_data.tsv" \
    -d  "$INPUT_DIR/ST000006_design.tsv" \
    -id Retention_Index \
    -g White_wine_type_and_source \
    -s "$OUTPUT_DIR/ST000006_kruskal_wallis_with_group_summary.tsv" \
    -f "$OUTPUT_DIR/ST000006_kruskal_wallis_with_group_flags.tsv" \
    -v "$OUTPUT_DIR/ST000006_kruskal_wallis_with_group_volcano.pdf"

: '
 The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="input"    value="ST000006_data.tsv"/>
        <param name="design"   value="ST000006_design.tsv"/>
        <param name="uniqueID" value="Retention_Index" />
        <param name="group"    value="White_wine_type_and_source" />
        <output name="summaries" file="ST000006_kruskal_wallis_with_group_summary.tsv" />
        <output name="flags"     file="ST000006_kruskal_wallis_with_group_flags.tsv" />
        <output name="volcano"   file="ST000006_kruskal_wallis_with_group_volcano.pdf" compare="sim_size" delta="10000" />
     </test>
    </tests>
'
