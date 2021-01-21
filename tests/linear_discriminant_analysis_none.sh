#!/bin/bash
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

CODEDIR="${SCRIPT_DIR}/.."
INPUTDIR="${SCRIPT_DIR}/../test-data"
OUTPUTDIR="$(pwd)/output_${TIMESTAMP}"
mkdir -p ${OUTPUTDIR}

/linear_discriminant_analysis.py \
		-i  $INPUTDIR/ST000006_data.tsv \
		-d  $INPUTDIR/ST000006_design.tsv \
		-id Retention_Index \
                -g White_wine_type_and_source \
		-cv none \
		-nc 2 \
         	-o   $OUTPUTDIR/ST000006_linear_discriminant_analysis_none_scores.tsv \
	        -oc  $OUTPUTDIR/ST000006_linear_discriminant_analysis_none_classification.tsv \
         	-oca $OUTPUTDIR/ST000006_linear_discriminant_analysis_none_classification_accuracy.tsv \
         	-f   $OUTPUTDIR/ST000006_linear_discriminant_analysis_none_figure.pdf


: '
 The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="input"            value="ST000006_data.tsv"/>
        <param name="design"           value="ST000006_design.tsv"/>
        <param name="uniqID"           value="Retention_Index" />
        <param name="group"            value="White_wine_type_and_source" />
        <param name="cross_validation" value="none"/>
        <param name="nComponents"      value="2"/>
        <output name="out"                       file="ST000006_linear_discriminant_analysis_none_scores.tsv" />
        <output name="outClassification"         file="ST000006_linear_discriminant_analysis_none_classification.tsv" />
        <output name="outClassificationAccuracy" file="ST000006_linear_discriminant_analysis_none_classification_accuracy.tsv" />
        <output name="figure"                    file="ST000006_linear_discriminant_analysis_none_figure.pdf" compare="sim_size" delta="10000"/>
     </test>
    </tests>

'
