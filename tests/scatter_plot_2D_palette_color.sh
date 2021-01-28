#!/bin/bash
#     <test>
#        <param name="input"   value="ST000006_principal_component_analysis_score_out.tsv"/>
#        <param name="design"  value="ST000006_design_group_name_underscore.tsv"/>
#        <param name="uniqID"  value="sampleID" />
#        <param name="group"   value="White_wine_type_and_source" />
#        <param name="x"       value="PC1" />
#        <param name="y"       value="PC2" />
#        <param name="palette" value="sequential" />
#        <param name="color"   value="Blues_3" />
#        <output name="figure" file="ST000006_scatter_plot_2D_palette_color_figure.pdf" compare="sim_size" delta="10000" />
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

scatter_plot_2D.py \
    -i  "$INPUT_DIR/ST000006_principal_component_analysis_score_out.tsv" \
    -d  "$INPUT_DIR/ST000006_design_group_name_underscore.tsv" \
    -id "sampleID" \
    -g  "White_wine_type_and_source" \
    -x  "PC1" \
    -y  "PC2" \
    -pal "sequential" \
    -col "Blues_3" \
    -f   "$OUTPUT_DIR/ST000006_scatter_plot_2D_palette_color_figure.pdf"

echo "### Finished test: ${TEST} on $(date)"
