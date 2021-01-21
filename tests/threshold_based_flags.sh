#!/bin/bash
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

CODEDIR="${SCRIPT_DIR}/.."
INPUTDIR="${SCRIPT_DIR}/../test-data"
OUTPUTDIR="$(pwd)/output_${TIMESTAMP}"
mkdir -p ${OUTPUTDIR}

/threshold_based_flags.py \
		-i  $INPUTDIR/ST000006_data.tsv \
		-d  $INPUTDIR/ST000006_design.tsv \
		-id Retention_Index \
                -g White_wine_type_and_source \
                -c 3000 \
         	-o  $OUTPUTDIR/ST000006_threshold_based_flags_output.tsv \


: '

The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="input"   value="ST000006_data.tsv"/>
        <param name="design"  value="ST000006_design.tsv"/>
        <param name="uniqID"  value="Retention_Index" />
        <param name="group"   value="White_wine_type_and_source" />
        <param name="cutoff"  value="3000" />
        <output name="output" file="ST000006_threshold_based_flags_output.tsv" />
     </test>
    </tests>

'

