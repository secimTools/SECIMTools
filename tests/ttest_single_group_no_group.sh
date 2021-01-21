#!/bin/bash
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

CODEDIR="${SCRIPT_DIR}/.."
INPUTDIR="${SCRIPT_DIR}/../test-data"
OUTPUTDIR="$(pwd)/output_${TIMESTAMP}"
mkdir -p ${OUTPUTDIR}

/ttest_single_group.py \
		-i  $INPUTDIR/ST000006_data.tsv \
		-d  $INPUTDIR/ST000006_design.tsv \
		-id Retention_Index \
                -mu 0 \
         	-s $OUTPUTDIR/ST000006_ttest_single_group_no_group_summary.tsv \
	        -f $OUTPUTDIR/ST000006_ttest_single_group_no_group_flags.tsv \
         	-v $OUTPUTDIR/ST000006_ttest_single_group_no_group_volcano.pdf




: '

The piece of code below is the test for the xml file.

    <tests>
      <test>
        <param name="input"      value="ST000006_data.tsv"/>
        <param name="design"     value="ST000006_design.tsv"/>
        <param name="uniqueID"   value="Retention_Index" />
        <param name="mu"         value="0" />
        <output name="summaries" file="ST000006_ttest_single_group_no_group_summary.tsv" />
        <output name="flags"     file="ST000006_ttest_single_group_no_group_flags.tsv" />
        <output name="volcano"   file="ST000006_ttest_single_group_no_group_volcano.pdf" compare="sim_size" delta="10000" />
      </test>
    </tests>

'




