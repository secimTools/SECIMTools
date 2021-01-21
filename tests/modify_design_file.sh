#!/bin/bash
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

CODEDIR="${SCRIPT_DIR}/.."
INPUTDIR="${SCRIPT_DIR}/../test-data"
OUTPUTDIR="$(pwd)/output_${TIMESTAMP}"
mkdir -p ${OUTPUTDIR}

/modify_design_file.py \
		-i  $INPUTDIR/ST000006_data.tsv \
		-d  $INPUTDIR/ST000006_design_group_name_underscore.tsv \
		-id Retention_Index \
                -g White_wine_type_and_source_nosp \
                -dp "Chardonnay_Napa_CA2003,Riesling_CA2004,SauvignonBlanc_LakeCounty_CA2001" \
         	-o  $OUTPUTDIR/ST000006_modify_design_file_output.tsv \


: '

The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="input"  value="ST000006_data.tsv"/>
        <param name="design" value="ST000006_design_group_name_underscore.tsv"/>
        <param name="uniqID" value="Retention_Index" />
        <param name="group"  value="White_wine_type_and_source_nosp" />
        <param name="toDrop" value="Chardonnay_Napa_CA2003,Riesling_CA2004,SauvignonBlanc_LakeCounty_CA2001" />
        <output name="out"   file="ST000006_modify_design_file_output.tsv" />
     </test>
    </tests>

'

