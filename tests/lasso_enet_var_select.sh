#!/bin/bash
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

CODEDIR="${SCRIPT_DIR}/.."
INPUTDIR="${SCRIPT_DIR}/../test-data"
OUTPUTDIR="$(pwd)/output_${TIMESTAMP}"
mkdir -p ${OUTPUTDIR}

/lasso_enet_var_select.py \
		-i  $INPUTDIR/ST000006_data.tsv \
		-d  $INPUTDIR/ST000006_design.tsv \
		-id Retention_Index \
                -g White_wine_type_and_source \
                -a 0.5 \
         	-c  $OUTPUTDIR/ST000006_lasso_enet_var_select_coefficients.tsv \
         	-f  $OUTPUTDIR/ST000006_lasso_enet_var_select_flags.tsv \
         	-p  $OUTPUTDIR/ST000006_lasso_enet_var_select_plots.pdf 



: '
 The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="input"  value="ST000006_data.tsv"/>
        <param name="design" value="ST000006_design.tsv"/>
        <param name="uniqID" value="Retention_Index" />
        <param name="group"  value="White_wine_type_and_source" />
        <param name="alpha"  value="0.5" />
        <output name="coefficients" file="ST000006_lasso_enet_var_select_coefficients.tsv" />
        <output name="flags"        file="ST000006_lasso_enet_var_select_flags.tsv" />
        <output name="plots"        file="ST000006_lasso_enet_var_select_plots.pdf" compare="sim_size" delta="10000" />
     </test>
    </tests>

'

