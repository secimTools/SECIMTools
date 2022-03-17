#!/bin/bash
#        <test>
#            <param name="wide"  value="upset_plot_input.tsv"/>
#            <param name="design" value="upset_plot_design.tsv"/>
#            <param name="uniqID" value="geneID" />
#            <param name="title"  value="upsetPlot_flags" />
#            <param name="outD" value="upsetPlot_flags" />
#            <param name="minSS" value="2" />
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

upset_plot.py \
    --wide "$INPUT_DIR/upset_plot_input.tsv" \
    --design "$INPUT_DIR/upset_plot_design.tsv" \
    --uniqID geneID \
    --title upsetPlot_flags \
    --outD upsetPlot_flags \
    --minSS 2

echo "### Finished test: ${TEST} on $(date)"
