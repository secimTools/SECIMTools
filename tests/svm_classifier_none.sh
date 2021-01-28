#!/bin/bash
#     <test>
#        <param name="train_wide"       value="ST000006_data.tsv"/>
#        <param name="train_design"     value="ST000006_design.tsv"/>
#        <param name="test_wide"        value="ST000006_data.tsv"/>
#        <param name="test_design"      value="ST000006_design.tsv"/>
#        <param name="group"            value="White_wine_type_and_source" />
#        <param name="uniqID"           value="Retention_Index" />
#        <param name="kernel"           value="linear"/>
#        <param name="degree"           value="3"/>
#        <param name="cross_validation" value="none"/>
#        <param name="C"                value="1"/>
#        <param name="C_lower_bound"    value="0.1"/>
#        <param name="C_upper_bound"    value="2"/>
#        <param name="a"                value="1"/>
#        <param name="b"                value="1"/>
#        <output name="outClassification"         file="ST000006_svm_classifier_train_classification.tsv" />
#        <output name="outClassificationAccuracy" file="ST000006_svm_classifier_train_classification_accuracy.tsv" />
#        <output name="outPrediction"             file="ST000006_svm_classifier_target_classification.tsv" />
#        <output name="outPredictionAccuracy"     file="ST000006_svm_classifier_target_classification_accuracy.tsv" />
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

svm_classifier.py \
    -trw  "$INPUT_DIR/ST000006_data.tsv" \
    -trd  "$INPUT_DIR/ST000006_design.tsv" \
    -tew  "$INPUT_DIR/ST000006_data.tsv" \
    -ted  "$INPUT_DIR/ST000006_design.tsv" \
    -id Retention_Index \
    -g White_wine_type_and_source \
    -k linear \
    -d 3 \
    -c 1 \
    -cv none \
    -a 1 \
    -b 1 \
    -oc  "$OUTPUT_DIR/ST000006_svm_classifier_train_classification.tsv" \
    -oca "$OUTPUT_DIR/ST000006_svm_classifier_train_classification_accuracy.tsv" \
    -op  "$OUTPUT_DIR/ST000006_svm_classifier_target_classification.tsv" \
    -opa "$OUTPUT_DIR/ST000006_svm_classifier_target_classification_accuracy.tsv" 

echo "### Finished test: ${TEST} on $(date)"
