#!/bin/bash



export PATH=/TB14/TB14/galaxy_27oct21/galaxy/database/dependencies/_conda/envs/__secimtools@21.2.21/bin:$PATH

PROJ=/nfshome/ammorse/mclab/SHARE/McIntyre_Lab/CID/sweet16
SCRIPTS=$PROJ/scripts/NMR/meta_analysis

OUT=$PROJ/nmr/roz_meta_analysis_comparisons
    mkdir -p $OUT

for PATHWAY in TCA UGT NI
do

if [[ ${PATHWAY} == "TCA" ]]
then
    MUTS=$(echo "KJ550 RB2347 VC1265 AUM2073 VC2524")
    echo " mutants are: $MUTS "
elif [[ ${PATHWAY} == "UGT" ]]
then
    MUTS=$(echo "RB2055 RB2550 RB2011 UGT49 UGT60")
elif [[ ${PATHWAY} == "NI" ]]
then
    MUTS=$(echo "CB4856 CX11314 DL238 N2")
fi

python $SCRIPTS/metaInSet_vs_metaPathway_addArgs.py \
    --in1 $PROJ/nmr/roz_meta_analysis_pathway/cdcl3_NI_summary.csv \
    --inPath $PATHWAY \
    --inMut $MUTS \
    --in2 $PROJ/nmr/roz_meta_analysis_inSet \
    --outD $OUT

done

#        if group == 'TCA':
#            muts = ["KJ550", "RB2347", "VC1265", "AUM2073", "VC2524"]
#        elif group == 'UGT':
#            muts = ["RB2055", "RB2550", "RB2011", 'UGT49', 'UGT60']
#        elif group == 'NI':
#            muts = ["CB4856", "CX11314", "DL238", "N2"]
