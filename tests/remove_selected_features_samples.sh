#!/bin/bash
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

CODEDIR="${SCRIPT_DIR}/.."
INPUTDIR="${SCRIPT_DIR}/../test-data"
OUTPUTDIR="$(pwd)/output_${TIMESTAMP}"
mkdir -p ${OUTPUTDIR}

/remove_selected_features_samples.py \
		-i  $INPUTDIR/ST000006_data.tsv \
		-d  $INPUTDIR/ST000006_design_group_name_underscore.tsv \
		-id Retention_Index \
		-f  $INPUTDIR/ST000006_kruskal_wallis_with_group_flags.tsv \
		-fft row \
		-fid Retention_Index \
		-fd  flag_significant_0p10_on_all_groups \
		-val 0 \
		-con 0 \
         	-ow $OUTPUTDIR/ST000006_remove_selected_features_samples_wide.tsv \
         	-of $OUTPUTDIR/ST000006_remove_selected_features_samples_flags.tsv  


: '

 The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="input"       value="ST000006_data.tsv"/>
        <param name="design"      value="ST000006_design.tsv"/>
        <param name="uniqID"      value="Retention_Index" />
        <param name="flags"       value="ST000006_kruskal_wallis_with_group_flags.tsv"/>
        <param name="typeofdrop"  value="row"/>
        <param name="flagUniqID"  value="Retention_Index" />
        <param name="flagToDrop"  value="flag_significant_0p10_on_all_groups" />
        <param name="reference"   value="0" />
        <param name="conditional" value="0" />
        <output name="outWide"    file="ST000006_remove_selected_features_samples_wide.tsv" />
        <output name="outFlags"   file="ST000006_remove_selected_features_samples_flags.tsv" />
     </test>
    </tests>

'






