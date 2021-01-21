#!/bin/bash
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

CODEDIR="${SCRIPT_DIR}/.."
INPUTDIR="${SCRIPT_DIR}/../test-data"
OUTPUTDIR="$(pwd)/output_${TIMESTAMP}"
mkdir -p ${OUTPUTDIR}

/modulated_modularity_clustering.py \
		-i  $INPUTDIR/ST000006_data.tsv \
		-d  $INPUTDIR/ST000006_design.tsv \
		-id Retention_Index \
                -c pearson \
         	-o $OUTPUTDIR/ST000006_modulated_modularity_clustering_out.tsv \
	        -f $OUTPUTDIR/ST000006_modulated_modularity_clustering_figure.pdf 







: '

The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="input"   value="ST000006_data.tsv"/>
        <param name="design"  value="ST000006_design.tsv"/>
        <param name="uniqID"  value="Retention_Index" />
        <param name="corr"    value="pearson" />
        <output name="output" file="ST000006_modulated_modularity_clustering_out.tsv" compare="sim_size" delta="10000"/>
        <output name="figure" file="ST000006_modulated_modularity_clustering_figure.pdf" compare="sim_size" delta="10000" />
     </test>
    </tests>

'



