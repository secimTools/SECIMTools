#!/bin/bash
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

CODEDIR="${SCRIPT_DIR}/.."
INPUTDIR="${SCRIPT_DIR}/../test-data"
OUTPUTDIR="$(pwd)/output_${TIMESTAMP}"
mkdir -p ${OUTPUTDIR}

/principal_component_analysis.py \
		-i  $INPUTDIR/ST000006_data.tsv \
		-d  $INPUTDIR/ST000006_design.tsv \
		-id Retention_Index \
                -g White_wine_type_and_source \
         	-lo $OUTPUTDIR/ST000006_principal_component_analysis_load_out.tsv \
	        -so $OUTPUTDIR/ST000006_principal_component_analysis_score_out.tsv \
         	-su $OUTPUTDIR/ST000006_principal_component_analysis_summary_out.tsv \
         	-f  $OUTPUTDIR/ST000006_principal_component_analysis_figure.pdf



: '

The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="input"  value="ST000006_data.tsv"/>
        <param name="design" value="ST000006_design.tsv"/>
        <param name="uniqID" value="Retention_Index" />
        <param name="group"  value="White_wine_type_and_source" />
        <output name="loadings" file="ST000006_principal_component_analysis_load_out.tsv" />
        <output name="scores"   file="ST000006_principal_component_analysis_score_out.tsv" />
        <output name="summary"  file="ST000006_principal_component_analysis_summary_out.tsv" />
        <output name="figures"  file="ST000006_principal_component_analysis_figure.pdf" compare="sim_size" delta="10000"/>
     </test>
    </tests>

'

