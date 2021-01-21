#!/bin/bash
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

CODEDIR="${SCRIPT_DIR}/.."
INPUTDIR="${SCRIPT_DIR}/../test-data"
OUTPUTDIR="$(pwd)/output_${TIMESTAMP}"
mkdir -p ${OUTPUTDIR}

/mahalanobis_distance.py \
		-i  $INPUTDIR/ST000006_data.tsv \
		-d  $INPUTDIR/ST000006_design.tsv \
		-id Retention_Index \
                -g  White_wine_type_and_source \
		-pen 0.5 \
         	-m  $OUTPUTDIR/ST000006_mahalanobis_distance_to_mean.tsv \
         	-f  $OUTPUTDIR/ST000006_mahalanobis_distance_figure.pdf \
         	-pw $OUTPUTDIR/ST000006_mahalanobis_distance_pairwise.tsv   


: '

 The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="input"   value="ST000006_data.tsv"/>
        <param name="design"  value="ST000006_design.tsv"/>
        <param name="uniqID"  value="Retention_Index" />
        <param name="group"   value="White_wine_type_and_source" />
        <param name="penalty" value="0.5" />
        <output name="plot"   file="ST000006_mahalanobis_distance_figure.pdf" compare="sim_size" delta="10000" />
        <output name="out1"   file="ST000006_mahalanobis_distance_to_mean.tsv" />
        <output name="out2"   file="ST000006_mahalanobis_distance_pairwise.tsv" />
     </test>
    </tests>

'

