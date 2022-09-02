#!/bin/bash
#      <test>
#        <param name="input"          value="ST000006_data.tsv"/>
#        <param name="design"         value="ST000006_design.tsv"/>
#        <param name="uniqID"         value="Retention_Index" />
#        <param name="transformation" value="log" />
#        <param name="log_base"       value="log" />
#        <param name="lambda_value"   value="0" />
#        <output name="oname"         file="ST000006_log_and_glog_transformation_log.tsv" />
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
    -t log \
    -l log10 \
    -o "$OUTPUT_DIR/ST000006_log_and_glog_transformation_log10.tsv"

echo "### Finished test: ${TEST} on $(date)"
