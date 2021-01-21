#!/bin/bash
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

CODEDIR="${SCRIPT_DIR}/.."
INPUTDIR="${SCRIPT_DIR}/../test-data"
OUTPUTDIR="$(pwd)/output_${TIMESTAMP}"
mkdir -p ${OUTPUTDIR}

/scatter_plot_3D.py \
		-i  $INPUTDIR/ST000006_principal_component_analysis_score_out.tsv \
		-d  $INPUTDIR/ST000006_design_group_name_underscore.tsv \
		-id sampleID \
                -g White_wine_type_and_source \
                -x PC1 \
                -y PC2 \
                -z PC3 \
		-r 30 \
		-r 23 \
                -pal sequential \
                -col Blues_3 \
	        -f $OUTPUTDIR/ST000006_scatter_plot_3D_palette_color_figure.pdf 




: '

The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="input"     value="ST000006_principal_component_analysis_score_out.tsv"/>
        <param name="design"    value="ST000006_design_group_name_underscore.tsv"/>
        <param name="uniqID"    value="sampleID" />
        <param name="group"     value="White_wine_type_and_source" />
        <param name="x"         value="PC1" />
        <param name="y"         value="PC2" />
        <param name="z"         value="PC3" />
        <param name="rotation"  value="30" />
        <param name="elevation" value="23" />
        <param name="palette"   value="sequential" />
        <param name="color"     value="Blues_3" />
        <output name="figure"   file="ST000006_scatter_plot_3D_palette_color_figure.pdf" compare="sim_size" delta="10000" />
     </test>
    </tests>

'


