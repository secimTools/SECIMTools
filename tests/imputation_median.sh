#!/bin/bash
#     <test>
#        <param name="input"      value="ST000006_data.tsv"/>
#        <param name="design"     value="ST000006_design.tsv"/>
#        <param name="uniqID"     value="Retention_Index" />
#        <param name="group"      value="White_wine_type_and_source" />
#        <param name="imputation" value="knn" />
#        <output name="imputed"   file="ST000006_imputation_output_KNN.tsv" />
#     </test>

## options
	## knn
	## bayesian
 	## mean
	## median


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

imputation.py \
    -i  "$INPUT_DIR/ST000006_data.tsv" \
    -d  "$INPUT_DIR/ST000006_design.tsv" \
    -id Retention_Index \
    -g White_wine_type_and_source \
    -s median \
    -o  "$OUTPUT_DIR/ST000006_imputation_output_MEDIAN.tsv"

echo "### Finished test: ${TEST} on $(date)"
