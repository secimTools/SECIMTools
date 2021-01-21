#!/bin/bash
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

CODEDIR="${SCRIPT_DIR}/.."
INPUTDIR="${SCRIPT_DIR}/../test-data"
OUTPUTDIR="$(pwd)/output_${TIMESTAMP}"
mkdir -p ${OUTPUTDIR}

/summarize_flags.py \
		-f  $INPUTDIR/ST000006_lasso_enet_var_select_flags.tsv \
		-id Retention_Index \
         	-os $OUTPUTDIR/ST000006_summarize_flags_outSummary.tsv \


: '

The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="flags"      value="ST000006_lasso_enet_var_select_flags.tsv"/>
        <param name="uniqID"     value="Retention_Index" />
        <output name="output"    file="ST000006_summarize_flags_outSummary.tsv" />
     </test>
    </tests>

'

