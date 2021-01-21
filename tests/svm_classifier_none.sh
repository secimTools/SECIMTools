#!/bin/bash
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

CODEDIR="${SCRIPT_DIR}/.."
INPUTDIR="${SCRIPT_DIR}/../test-data"
OUTPUTDIR="$(pwd)/output_${TIMESTAMP}"
mkdir -p ${OUTPUTDIR}

/svm_classifier.py \
		-trw  $INPUTDIR/ST000006_data.tsv \
		-trd  $INPUTDIR/ST000006_design.tsv \
		-tew  $INPUTDIR/ST000006_data.tsv \
		-ted  $INPUTDIR/ST000006_design.tsv \
		-id Retention_Index \
                -g White_wine_type_and_source \
		-k linear \
		-d 3 \
		-c 1 \
		-cv none \
		-a 1 \
		-b 1 \
	        -oc  $OUTPUTDIR/ST000006_svm_classifier_train_classification.tsv \
         	-oca $OUTPUTDIR/ST000006_svm_classifier_train_classification_accuracy.tsv \
	        -op  $OUTPUTDIR/ST000006_svm_classifier_target_classification.tsv \
         	-opa $OUTPUTDIR/ST000006_svm_classifier_target_classification_accuracy.tsv 






: '

The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="train_wide"       value="ST000006_data.tsv"/>
        <param name="train_design"     value="ST000006_design.tsv"/>
        <param name="test_wide"        value="ST000006_data.tsv"/>
        <param name="test_design"      value="ST000006_design.tsv"/>
        <param name="group"            value="White_wine_type_and_source" />
        <param name="uniqID"           value="Retention_Index" />
        <param name="kernel"           value="linear"/>
        <param name="degree"           value="3"/>
        <param name="cross_validation" value="none"/>
        <param name="C"                value="1"/>
        <param name="C_lower_bound"    value="0.1"/>
        <param name="C_upper_bound"    value="2"/>
        <param name="a"                value="1"/>
        <param name="b"                value="1"/>
        <output name="outClassification"         file="ST000006_svm_classifier_train_classification.tsv" />
        <output name="outClassificationAccuracy" file="ST000006_svm_classifier_train_classification_accuracy.tsv" />
        <output name="outPrediction"             file="ST000006_svm_classifier_target_classification.tsv" />
        <output name="outPredictionAccuracy"     file="ST000006_svm_classifier_target_classification_accuracy.tsv" />
     </test>
    </tests>

'




