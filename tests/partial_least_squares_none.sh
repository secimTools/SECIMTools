#!/bin/bash
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

CODEDIR="${SCRIPT_DIR}/.."
INPUTDIR="${SCRIPT_DIR}/../test-data"
OUTPUTDIR="$(pwd)/output_${TIMESTAMP}"
mkdir -p ${OUTPUTDIR}

/partial_least_squares.py \
		-i  $INPUTDIR/ST000006_data.tsv \
		-d  $INPUTDIR/ST000006_design_group_name_underscore.tsv \
		-id Retention_Index \
                -g White_wine_type_and_source \
		-t "Chardonnay_ Napa_ CA 2003,Riesling_ CA 2004" \
		-cv none \
		-n 2 \
         	-os  $OUTPUTDIR/ST000006_partial_least_squares_none_scores.tsv \
         	-ow  $OUTPUTDIR/ST000006_partial_least_squares_none_weights.tsv \
	        -oc  $OUTPUTDIR/ST000006_partial_least_squares_none_classification.tsv \
         	-oca $OUTPUTDIR/ST000006_partial_least_squares_none_classification_accuracy.tsv \
         	-f   $OUTPUTDIR/ST000006_partial_least_squares_none_figure.pdf 




: '

 The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="input"             value="ST000006_data.tsv"/>
        <param name="design"            value="ST000006_design_group_name_underscore.tsv"/>
        <param name="uniqID"            value="Retention_Index" />
        <param name="group"             value = "White_wine_type_and_source" />
        <param name="toCompare"         value="Chardonnay_ Napa_ CA 2003,Riesling_ CA 2004" />
        <param name="cross_validation"  value="none"/>
        <param name="nComp" type="text" value="2"/>
        <output name="outScores"                 file="ST000006_partial_least_squares_none_scores.tsv" />
        <output name="outWeights"                file="ST000006_partial_least_squares_none_weights.tsv" />
        <output name="outClassification"         file="ST000006_partial_least_squares_none_classification.tsv" />
        <output name="outClassificationAccuracy" file="ST000006_partial_least_squares_none_classification_accuracy.tsv" />
        <output name="figures"                   file="ST000006_partial_least_squares_none_figure.pdf" compare="sim_size" delta="10000" />
     </test>
    </tests>

'




