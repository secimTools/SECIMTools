#!/bin/bash
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

CODEDIR="${SCRIPT_DIR}/.."
INPUTDIR="${SCRIPT_DIR}/../test-data"
OUTPUTDIR="$(pwd)/output_${TIMESTAMP}"
mkdir -p ${OUTPUTDIR}

/run_order_regression.py \
		-i  $INPUTDIR/ST000006_data.tsv \
		-d  $INPUTDIR/ST000006_design.tsv \
		-id Retention_Index \
                -g White_wine_type_and_source \
                -o run_Order_fake_variable \
         	-f  $OUTPUTDIR/ST000006_run_order_regression_figure.pdf \
	        -t  $OUTPUTDIR/ST000006_run_order_regression_table.tsv \
         	-fl $OUTPUTDIR/ST000006_run_order_regression_flags.tsv


: '

The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="input"  value="ST000006_data.tsv"/>
        <param name="design" value="ST000006_design.tsv"/>
        <param name="uniqID" value="Retention_Index" />
        <param name="group"  value="White_wine_type_and_source" />
        <param name="order"  value="run_Order_fake_variable" />
        <output name="figure"        file="ST000006_run_order_regression_figure.pdf" compare="sim_size" delta="10000" />
        <output name="order_summary" file="ST000006_run_order_regression_table.tsv" />
        <output name="flags"         file="ST000006_run_order_regression_flags.tsv" />
     </test>
    </tests>

'

