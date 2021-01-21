#!/bin/bash
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

CODEDIR="${SCRIPT_DIR}/.."
INPUTDIR="${SCRIPT_DIR}/../test-data"
OUTPUTDIR="$(pwd)/output_${TIMESTAMP}"
mkdir -p ${OUTPUTDIR}

/retention_time_flags.py \
		-i  $INPUTDIR/TEST0000_rt.tsv \
		-d  $INPUTDIR/TEST0000_design.tsv \
		-id rowID \
         	-f  $OUTPUTDIR/TEST0000_retention_time_flags_figure.pdf \
         	-fl $OUTPUTDIR/TEST0000_retention_time_flags_flag.tsv  


: '

 The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="input"   value="TEST0000_rt.tsv"/>
        <param name="design"  value="TEST0000_design.tsv"/>
        <param name="uniqID"  value="rowID" />
        <output name="RTplot" file="TEST0000_retention_time_flags_figure.pdf" compare="sim_size" delta="10000" />
        <output name="RTflag" file="TEST0000_retention_time_flags_flag.tsv" />
     </test>
    </tests>

'


