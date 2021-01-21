#!/bin/bash
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

CODEDIR="${SCRIPT_DIR}/.."
INPUTDIR="${SCRIPT_DIR}/../test-data"
OUTPUTDIR="$(pwd)/output_${TIMESTAMP}"
mkdir -p ${OUTPUTDIR}

# /magnitude_difference_flags.py \
/magnitude_difference_flags.py \
		-i  $INPUTDIR/ST000006_data.tsv \
		-d  $INPUTDIR/ST000006_design.tsv \
		-id Retention_Index \
                -nz \
                -g White_wine_type_and_source \
         	-f  $OUTPUTDIR/ST000006_magnitude_difference_flags_figure.pdf \
	        -fl $OUTPUTDIR/ST000006_magnitude_difference_flags_flags.tsv \
	        -c  $OUTPUTDIR/ST000006_magnitude_difference_flags_counts.tsv 





: '

 The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="input"   value="ST000006_data.tsv"/>
        <param name="design"  value="ST000006_design.tsv"/>
        <param name="uniqID"  value="Retention_Index" />
        <param name="group"   value="White_wine_type_and_source" />
        <param name="nonzero" value="yes" />
        <output name="figure" file="ST000006_magnitude_difference_flags_figure.pdf" compare="sim_size" delta="10000"/>
        <output name="flags"  file="ST000006_magnitude_difference_flags_flags.tsv" />
     </test>
    </tests>

'


