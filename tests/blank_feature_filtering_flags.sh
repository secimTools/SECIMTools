#!/bin/bash
SCRIPT=$(basename "${BASH_SOURCE[0]}"); NAME="${SCRIPT%.*}"
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
INPUT_DIR="${SCRIPT_DIR}/../test-data"
OUTPUT_DIR="$(pwd)/output/${NAME}_${TIMESTAMP}"
if [[ $# -gt 0 ]]; then OUTPUT_DIR=$1 ; fi
mkdir -p "${OUTPUT_DIR}"

blank_feature_filtering_flags.py \
    -i  "$INPUT_DIR/TEST0000_data.tsv" \
    -d  "$INPUT_DIR/TEST0000_design.tsv" \
    -id rowID \
    -g  group_combined \
    -bv 5000 \
    -bn blank \
    -c  100 \
    -f  "$OUTPUT_DIR/TEST0000_blank_feature_filtering_flags_outflags.tsv" \
    -b  "$OUTPUT_DIR/TEST0000_blank_feature_filtering_flags_outbff.tsv"
: '

 The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="input"  value="TEST0000_data.tsv"/>
        <param name="design" value="TEST0000_design.tsv"/>
        <param name="uniqID" value="rowID" />
        <param name="group"  value="group_combined" />
        <param name="blank"  value="blank" />
        <param name="bff"    value="5000" />
        <param name="cv"     value="100" />
        <output name="outflags" file="TEST0000_blank_feature_filtering_flags_outflags.tsv" />
        <output name="outbff"   file="TEST0000_blank_feature_filtering_flags_outbff.tsv" />
     </test>
    </tests>
'
