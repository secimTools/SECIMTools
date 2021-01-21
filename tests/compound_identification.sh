#!/bin/bash
SCRIPT=$(basename "${BASH_SOURCE[0]}"); NAME="${SCRIPT%.*}"
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
INPUT_DIR="${SCRIPT_DIR}/../test-data"
OUTPUT_DIR="$(pwd)/output/${NAME}_${TIMESTAMP}"
if [[ $# -gt 0 ]]; then OUTPUT_DIR=$1 ; fi
mkdir -p "${OUTPUT_DIR}"

compound_identification.py \
    -a   "$INPUT_DIR/TEST0000_mzrt_first.tsv" \
    -id rowID_first \
    -mzi MZ_first \
    -rti RT_first \
    -lib "$INPUT_DIR/TEST0000_database.tsv" \
    -lid name_database \
    -lmzi MZ_database \
    -lrti RT_database \
    -o  "$OUTPUT_DIR/TEST0000_compound_identification_output.tsv"

: '

 The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="anno"      value="TEST0000_mzrt_first.tsv"/>
        <param name="uniqID"    value="rowID_first"/>
        <param name="mzID"      value="MZ_first" />
        <param name="rtID"      value="RT_first" />
        <param name="library"   value="TEST0000_database.tsv"/>
        <param name="libuniqID" value="name_database"/>
        <param name="libmzID"   value="MZ_database" />
        <param name="librtID"   value="RT_database" />
        <output name="output"   file="TEST0000_compound_identification_output.tsv" />
     </test>
    </tests>

'
