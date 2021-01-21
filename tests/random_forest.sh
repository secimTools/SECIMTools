#!/bin/bash
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

CODEDIR="${SCRIPT_DIR}/.."
INPUTDIR="${SCRIPT_DIR}/../test-data"
OUTPUTDIR="$(pwd)/output_${TIMESTAMP}"
mkdir -p ${OUTPUTDIR}

/random_forest.py \
		-i  $INPUTDIR/ST000006_data.tsv \
		-d  $INPUTDIR/ST000006_design.tsv \
		-id Retention_Index \
                -g White_wine_type_and_source \
         	-o  $OUTPUTDIR/ST000006_random_forest_out.tsv \
	        -o2 $OUTPUTDIR/ST000006_random_forest_out2.tsv \
         	-f  $OUTPUTDIR/ST000006_random_forest_figure.pdf


: '

The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="input"  value="ST000006_data.tsv"/>
        <param name="design" value="ST000006_design.tsv"/>
        <param name="uniqID" value="Retention_Index" />
        <param name="group"  value="White_wine_type_and_source" />
        <output name="outfile1" file="ST000006_random_forest_out.tsv" />
        <output name="outfile2" file="ST000006_random_forest_out2.tsv" />
        <output name="figure"   file="ST000006_random_forest_figure.pdf" compare="sim_size" delta="10000"/>
     </test>
    </tests>

'

