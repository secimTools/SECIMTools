#!/bin/bash
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

CODEDIR="${SCRIPT_DIR}/.."
INPUTDIR="${SCRIPT_DIR}/../test-data"
OUTPUTDIR="$(pwd)/output_${TIMESTAMP}"
mkdir -p ${OUTPUTDIR}

/mzrt_match.py \
		-a1   $INPUTDIR/TEST0000_mzrt_first.tsv \
		-a2   $INPUTDIR/TEST0000_mzrt_second.tsv \
		-ID1 rowID_first \
		-ID2 rowID_second \
		-mz1 MZ_first \
		-rt1 RT_first \
		-mz2 MZ_second \
		-rt2 RT_second \
	        -a   $OUTPUTDIR/TEST0000_mzrt_match_all.tsv \
	        -m   $OUTPUTDIR/TEST0000_mzrt_match_matched.tsv \
	        -s   $OUTPUTDIR/TEST0000_mzrt_match_summary.tsv \
	        -u1  $OUTPUTDIR/TEST0000_mzrt_match_unmatched_first.tsv  \
	        -u2  $OUTPUTDIR/TEST0000_mzrt_match_unmatched_second.tsv \
	        -fig $OUTPUTDIR/TEST0000_mzrt_match_figure.pdf 



: '

 The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="anno1"    value="TEST0000_mzrt_first.tsv"/>
        <param name="anno2"    value="TEST0000_mzrt_second.tsv"/>
        <param name="uniqID1"  value="rowID_first"/>
        <param name="uniqID2"  value="rowID_second"/>
        <param name="mzID1"    value="MZ_first" />
        <param name="mzID2"    value="MZ_second" />
        <param name="rtID1"    value="RT_first" />
        <param name="rtID2"    value="RT_second" />
        <output name="all"        file="TEST0000_mzrt_match_all.tsv" />
        <output name="matched"    file="TEST0000_mzrt_match_matched.tsv" />
        <output name="unmatched1" file="TEST0000_mzrt_match_unmatched_first.tsv" />
        <output name="unmatched2" file="TEST0000_mzrt_match_unmatched_second.tsv" />
        <output name="summary"    file="TEST0000_mzrt_match_summary.tsv" />
        <output name="figure"     file="TEST0000_mzrt_match_figure.pdf" compare="sim_size" delta="10000" />
     </test>
    </tests>

'


