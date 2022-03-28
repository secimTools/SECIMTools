#!/bin/bash
#        <test>
#            <param name="input"  value="ready_cdcl3_rank_sbys.tsv"/>
#            <param name="design" value="dsgn_nmr_4_MA_cdcl3_set1_CB4856.tsv"/>
#            <param name="uniqID" value="ppm" />
#            <param name="study"  value="batch" />
#            <param name="treatment" value="genotype" />
#            <param name="contrast" value="PD1074,CB4856" />
#            <output name="report" file="meta_analysis_report.tsv" />
#            <output name="summary" file="meta_analysis_summary.csv" />
#        </test>

SCRIPT=$(basename "${BASH_SOURCE[0]}");
TEST="${SCRIPT%.*}"
echo "Echoing ${BASH_SOURCE[0]}"
TESTDIR="testout/${TEST}"
INPUT_DIR="galaxy/test-data"
OUTPUT_DIR=$TESTDIR
rm -rf "${TESTDIR}"
mkdir -p "${TESTDIR}"
echo "### Starting test: ${TEST}"
if [[ $# -gt 0 ]]; then OUTPUT_DIR=$1 ; fi
mkdir -p "${OUTPUT_DIR}"

meta_analysis.py \
    --wide   "$INPUT_DIR/ready_cdcl3_rank_sbys.tsv" \
    --design "$INPUT_DIR/dsgn_nmr_4_MA_cdcl3_set1_CB4856.tsv" \
    --uniqID ppm \
    --study  batch \
    --treatment genotype \
    --contrast PD1074,CB4856 \
    --model "FE" \
    --effectSize "SMD" \
    --cmMethod "UB" \
    --report "$OUTPUT_DIR/meta_analysis_report.tsv" \
    --summary "$OUTPUT_DIR/meta_analysis_report.tsv"

echo "### Finished test: ${TEST} on $(date)"
