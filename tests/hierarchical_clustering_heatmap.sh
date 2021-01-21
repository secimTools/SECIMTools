#!/bin/bash
SCRIPT=$(basename "${BASH_SOURCE[0]}"); NAME="${SCRIPT%.*}"
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
INPUT_DIR="${SCRIPT_DIR}/../test-data"
OUTPUT_DIR="$(pwd)/output/${NAME}_${TIMESTAMP}"
if [[ $# -gt 0 ]]; then OUTPUT_DIR=$1 ; fi
mkdir -p "${OUTPUT_DIR}"

hierarchical_clustering_heatmap.py \
    -i  "$INPUT_DIR/ST000006_data.tsv" \
    -d  "$INPUT_DIR/ST000006_design.tsv" \
    -id Retention_Index \
    -den \
    -l "x,y" \
    -f  "$OUTPUT_DIR/ST000006_hierarchical_clustering_heatmap_figure.pdf"

: '
 The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="input"     value="ST000006_data.tsv"/>
        <param name="design"    value="ST000006_design.tsv"/>
        <param name="uniqID"    value="Retention_Index" />
        <param name="dendogram" value="True" />
        <param name="labels"    value="x,y" />
        <output name="fig"      file="ST000006_hierarchical_clustering_heatmap_figure.pdf" compare="sim_size" delta="10000" />
     </test>
    </tests>
'
