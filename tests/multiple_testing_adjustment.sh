#!/bin/bash
TIMESTAMP=$(date +%s)
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

CODEDIR="${SCRIPT_DIR}/.."
INPUTDIR="${SCRIPT_DIR}/../test-data"
OUTPUTDIR="$(pwd)/output_${TIMESTAMP}"
mkdir -p ${OUTPUTDIR}

/multiple_testing_adjustment.py \
		-i  $INPUTDIR/ST000006_anova_fixed_with_group_summary.tsv \
		-id Retention_Index \
                -pv  "prob_greater_than_t_for_diff_Chardonnay, Carneros, CA 2003 (CH01)-Chardonnay, Carneros, CA 2003 (CH02)" \
		-a 0.05 \
         	-on $OUTPUTDIR/ST000006_multiple_testing_adjustment_outadjusted.tsv \
         	-fl $OUTPUTDIR/ST000006_multiple_testing_adjustment_flags.tsv 


: '

 The piece of code below is the test for the xml file.

    <tests>
     <test>
        <param name="input"  value="ST000006_anova_fixed_with_group_summary.tsv"/>
        <param name="uniqID" value="Retention_Index" />
        <param name="pval"   value="prob_greater_than_t_for_diff_Chardonnay, Carneros, CA 2003 (CH01)-Chardonnay, Carneros, CA 2003 (CH02)" />
        <param name="alpha"  value="0.05" />
        <output name="outadjusted" file="ST000006_multiple_testing_adjustment_outadjusted.tsv" />
        <output name="flags"       file="ST000006_multiple_testing_adjustment_flags.tsv" />
     </test>
    </tests>

'

