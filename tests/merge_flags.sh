#!/bin/bash
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

CODEDIR="${SCRIPT_DIR}/.."
INPUTDIR="${SCRIPT_DIR}/../test-data"
OUTPUTDIR="$(pwd)/output_${TIMESTAMP}"
mkdir -p ${OUTPUTDIR}

/merge_flags.py \
		-i  "$INPUTDIR/ST000006_run_order_regression_flags.tsv,$INPUTDIR/ST000006_lasso_enet_var_select_flags.tsv" \
		-f  "ST000006_run_order_regression_flags_tsv" "ST000006_lasso_enet_var_select_flags_tsv" \
		-fid Retention_Index \
         	-o  $OUTPUTDIR/ST000006_merge_flags_output.tsv \


: '

The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="input"      value="ST000006_run_order_regression_flags.tsv,ST000006_lasso_enet_var_select_flags.tsv"/>
        <param name="flagUniqID" value="Retention_Index" />
        <param name="filename"   value="ST000006_run_order_regression_flags ST000006_lasso_enet_var_select_flags" />
        <output name="output"    file="ST000006_merge_flags_output.tsv" />
     </test>
    </tests>

'

