#!/bin/bash
#     <test>
#        <param name="input"             value="ST000006_data.tsv"/>
#        <param name="design"            value="ST000006_design_group_name_underscore.tsv"/>
#        <param name="uniqID"            value="Retention_Index" />
#        <param name="group"             value = "White_wine_type_and_source" />
#        <param name="toCompare"         value="Chardonnay_ Napa_ CA 2003,Riesling_ CA 2004" />
#        <param name="cross_validation"  value="none"/>
#        <param name="nComp" type="text" value="2"/>
#        <output name="outScores"                 file="ST000006_partial_least_squares_none_scores.tsv" />
#        <output name="outWeights"                file="ST000006_partial_least_squares_none_weights.tsv" />
#        <output name="outClassification"         file="ST000006_partial_least_squares_none_classification.tsv" />
#        <output name="outClassificationAccuracy" file="ST000006_partial_least_squares_none_classification_accuracy.tsv" />
#        <output name="figures"                   file="ST000006_partial_least_squares_none_figure.pdf" compare="sim_size" delta="10000" />
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

partial_least_squares.py \
    -i   "$INPUT_DIR/ST000006_data.tsv" \
    -d   "$INPUT_DIR/ST000006_design_group_name_underscore.tsv" \
    -id  Retention_Index \
    -g   White_wine_type_and_source \
    -t   "Chardonnay_ Napa_ CA 2003,Riesling_ CA 2004" \
    -cv  none \
    -n   2 \
    -os  "$OUTPUT_DIR/ST000006_partial_least_squares_none_scores.tsv" \
    -ow  "$OUTPUT_DIR/ST000006_partial_least_squares_none_weights.tsv" \
    -oc  "$OUTPUT_DIR/ST000006_partial_least_squares_none_classification.tsv" \
    -oca "$OUTPUT_DIR/ST000006_partial_least_squares_none_classification_accuracy.tsv" \
    -f   "$OUTPUT_DIR/ST000006_partial_least_squares_none_figure.pdf"

echo "### Finished test: ${TEST} on $(date)"
