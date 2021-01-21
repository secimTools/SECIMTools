#!/bin/bash
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

CODEDIR="${SCRIPT_DIR}/.."
INPUTDIR="${SCRIPT_DIR}/../test-data"
OUTPUTDIR="$(pwd)/output_${TIMESTAMP}"
mkdir -p ${OUTPUTDIR}

/scatter_plot_2D.py \
		-i  $INPUTDIR/ST000006_principal_component_analysis_score_out.tsv \
		-d  $INPUTDIR/ST000006_design_group_name_underscore.tsv \
		-id sampleID \
                -g White_wine_type_and_source \
                -x PC1 \
                -y PC2 \
	        -f $OUTPUTDIR/ST000006_scatter_plot_2D_default_figure.pdf 






: '

The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="input"   value="ST000006_principal_component_analysis_score_out.tsv"/>
        <param name="design"  value="ST000006_design_group_name_underscore.tsv"/>
        <param name="uniqID"  value="sampleID" />
        <param name="group"   value="White_wine_type_and_source" />
        <param name="x"       value="PC1" />
        <param name="y"       value="PC2" />
        <output name="figure" file="ST000006_scatter_plot_2D_default_figure.pdf" compare="sim_size" delta="10000" />
     </test>
    </tests>

'

