#!/bin/bash
#     <test>
#        <param name="input"   value="TEST0000_rt.tsv"/>
#        <param name="design"  value="TEST0000_design.tsv"/>
#        <param name="uniqID"  value="rowID" />
#        <output name="RTplot" file="TEST0000_retention_time_flags_figure.pdf" compare="sim_size" delta="10000" />
#        <output name="RTflag" file="TEST0000_retention_time_flags_flag.tsv" />
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

retention_time_flags.py \
    -i  "$INPUT_DIR/TEST0000_rt.tsv" \
    -d  "$INPUT_DIR/TEST0000_design.tsv" \
    -id "rowID" \
    -f  "$OUTPUT_DIR/TEST0000_retention_time_flags_figure.pdf" \
    -fl "$OUTPUT_DIR/TEST0000_retention_time_flags_flag.tsv"

echo "### Finished test: ${TEST} on $(date)"
