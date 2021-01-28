#!/bin/bash
#      <test>
#        <param name="input"          value="ST000006_data.tsv"/>
#        <param name="design"         value="ST000006_design.tsv"/>
#        <param name="uniqID"         value="Retention_Index" />
#        <param name="transformation" value="glog" />
#        <param name="log_base"       value="log" />
#        <param name="lambda_value"   value="1000000" />
#        <output name="oname"         file="ST000006_log_and_glog_transformation_glog_lambda_1000000.tsv" />
#      </test>

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

log_and_glog_transformation.py \
        -i  "$INPUT_DIR/ST000006_data.tsv" \
        -d  "$INPUT_DIR/ST000006_design.tsv" \
        -id Retention_Index \
        -t glog \
        -l log \
        -la 1000000 \
        -o "$OUTPUT_DIR/ST000006_log_and_glog_transformation_glog_lambda_1000000.tsv"

echo "### Finished test: ${TEST} on $(date)"
