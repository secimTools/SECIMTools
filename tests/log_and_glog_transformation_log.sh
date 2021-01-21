#!/bin/bash
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

CODEDIR="${SCRIPT_DIR}/.."
INPUTDIR="${SCRIPT_DIR}/../test-data"
OUTPUTDIR="$(pwd)/output_${TIMESTAMP}"
mkdir -p ${OUTPUTDIR}

/log_and_glog_transformation.py \
		-i  $INPUTDIR/ST000006_data.tsv \
		-d  $INPUTDIR/ST000006_design.tsv \
		-id Retention_Index \
		-t log \
		-l log \
		-o $OUTPUTDIR/ST000006_log_and_glog_transformation_log.tsv 





: '
 The piece of code below is the test for the xml file.

    <tests>
      <test>
        <param name="input"          value="ST000006_data.tsv"/>
        <param name="design"         value="ST000006_design.tsv"/>
        <param name="uniqID"         value="Retention_Index" />
        <param name="transformation" value="log" />
        <param name="log_base"       value="log" />
        <param name="lambda_value"   value="0" />
        <output name="oname"         file="ST000006_log_and_glog_transformation_log.tsv" />
      </test>
    </tests>

'





