SCRIPTS=$PWD/..
DATA=$PWD/../test-data/infection_data.tsv
DESIGN=$PWD/../test-data/infection_design.tsv
OUTPUT=$PWD/infection_results

ID="Retention_Index"
GROUP="Xoo_infection" 
BLANK=""
RUNORDER=""
SUBSET="mock,no treatment"
SAMPLETOSUBSET="SA000272"

#ID="rowID"
#GROUP="Characteristics_organism" 
#BLANK=""
#RUNORDER=""
#SUBSET="Bos taurus,Ovis aries"
#SAMPLETOSUBSET="batch01_QC01"

#ID="Retention_Index"
#GROUP="White_wine_type_and_source" 
#BLANK=""
#RUNORDER=""
#SUBSET="Chardonnay  Carneros  CA 2003 (CH01),Chardonnay  Carneros  CA 2003 (CH02)"
#SAMPLETOSUBSET="Chardonnay  Carneros  CA 2003 (CH02)"

#ID="rowID"
#GROUP="group" 
#BLANK="blank"
#RUNORDER="runOrder"
#SUBSET="8C,8T"
#SAMPLETOSUBSET="ISTD_09"

# "Chardonnay  Carneros  CA 2003 (CH01),Chardonnay  Carneros  CA 2003 (CH02)"

python $SCRIPTS/anova_fixed.py \
	-i $DATA \
	-d $DESIGN \
	-id $ID \
	-f $GROUP \
	-t C \
	-o $OUTPUT/del_anova_table.tsv \
	-fl $OUTPUT/del_anova_flags.tsv \
	-f1 $OUTPUT/del_anova_qqplot.pdf \
	-f2 $OUTPUT/del_anova_volcano.pdf &

python $SCRIPTS/bland_altmant_plot.py \
	-i $DATA \
	-d $DESIGN \
	-id $ID \
	-g $GROUP \
	-f $OUTPUT/del_ba_baplots.pdf \
	-fd $OUTPUT/del_ba_distributions.pdf \
	-fs $OUTPUT/del_ba_flag_sample.tsv \
	-ff $OUTPUT/del_ba_flag_features.tsv &

python $SCRIPTS/blank_feature_filtering_flags.py \
	-i $DATA \
	-d $DESIGN \
	-id $ID \
	-g $GROUP \
	-bn $BLANK \
	-f $OUTPUT/del_bff_flags.tsv \
	-b $OUTPUT/del_bff_bff.tsv &

python $SCRIPTS/coefificient_variation_flags.py \
	-i $DATA \
	-d $DESIGN \
	-id $ID \
	-f $OUTPUT/del_cv_figure.pdf \
	-o $OUTPUT/del_cv_flag.tsv &

##############################################################
python $SCRIPTS/data_rescalling.py \
	-i $DATA \
	-d $DESIGN \
	-id $ID \
	-m "mean" \
	-o $OUTPUT/del_datarescalling_mean.tsv &

python $SCRIPTS/data_rescalling.py \
	-i $DATA \
	-d $DESIGN \
	-id $ID \
	-m "sum" \
	-o $OUTPUT/del_datarescalling_sum.tsv &

python $SCRIPTS/data_rescalling.py \
	-i $DATA \
	-d $DESIGN \
	-id $ID \
	-m "median" \
	-o $OUTPUT/del_datarescalling_median.tsv &

##############################################################

python $SCRIPTS/log_transformation.py \
	-i $DATA \
	-d $DESIGN \
	-id $ID \
	-l "log" \
	-o $OUTPUT/del_log_10.tsv

#############################################################
python $SCRIPTS/distribution_features.py \
	-i $OUTPUT/del_log_10.tsv \
	-d $DESIGN \
	-id $ID \
	-g $GROUP \
	-f $OUTPUT/del_distribution_features_groups.pdf

python $SCRIPTS/distribution_features.py \
	-i $OUTPUT/del_log_10.tsv \
	-d $DESIGN \
	-id $ID \
	-f $OUTPUT/del_distribution_features_noGroups.pdf

python $SCRIPTS/distribution_samples.py \
	-i $OUTPUT/del_log_10.tsv \
	-d $DESIGN \
	-id $ID \
	-g $GROUP \
	-f $OUTPUT/del_distribution_samples_groups.pdf

python $SCRIPTS/distribution_samples.py \
	-i $OUTPUT/del_log_10.tsv \
	-d $DESIGN \
	-id $ID \
	-f $OUTPUT/del_distribution_samples_noGroups.pdf 
#############################################################

python $SCRIPTS/hierarchical_clustering_heatmap.py \
	-i $OUTPUT/del_log_10.tsv \
	-d $DESIGN \
	-id $ID \
	-f $OUTPUT/del_hm_single.pdf

python $SCRIPTS/hierarchical_clustering_heatmap.py \
	-i $OUTPUT/del_log_10.tsv \
	-d $DESIGN \
	-id $ID \
	-f $OUTPUT/del_hm_denogram.pdf \
	-den

#############################################################
python $SCRIPTS/imputation.py \
	-i $DATA \
	-d $DESIGN \
	-id $ID \
	-g $GROUP \
	-o $OUTPUT/del_imputed_knn.tsv \
	-s knn &

python $SCRIPTS/imputation.py \
	-i $DATA \
	-d $DESIGN \
	-id $ID \
	-g $GROUP \
	-o $OUTPUT/del_imputed_bayesian.tsv \
	-s bayesian \
	-dist Normal &

python $SCRIPTS/imputation.py \
	-i $DATA \
	-d $DESIGN \
	-id $ID \
	-g $GROUP \
	-o $OUTPUT/del_imputed_bayesian.tsv \
	-s bayesian \
	-dist Poisson &

python $SCRIPTS/imputation.py \
	-i $DATA \
	-d $DESIGN \
	-id $ID \
	-g $GROUP \
	-o $OUTPUT/del_imputed_mean.tsv \
	-s mean &

python $SCRIPTS/imputation.py \
	-i $DATA \
	-d $DESIGN \
	-id $ID \
	-g $GROUP \
	-o $OUTPUT/del_imputed_median.tsv \
	-s median &
##############################################################

python $SCRIPTS/lasso_enet_var_select.py \
	-i $DATA \
	-d $DESIGN \
	-id $ID \
	-g $GROUP \
	-a 0.5 \
	-c $OUTPUT/del_lasso_coefficients.tsv \
	-f $OUTPUT/del_lasso_flags.tsv \
	-p $OUTPUT/del_lasso_plots.pdf &

python $SCRIPTS/linear_discriminant_analysis.py \
	-i $DATA \
	-d $DESIGN \
	-id $ID \
	-g $GROUP \
	-o $OUTPUT/del_lda_scores.tsv \
	-f $OUTPUT/del_lda_scatterplot.pdf &

python $SCRIPTS/magnitud_difference_flags.py \
	-i $DATA \
	-d $DESIGN \
	-id $ID \
	-g $GROUP \
	-fl $OUTPUT/del_magnitud_flags.tsv \
	-f $OUTPUT/del_magnitud_scatterplot.pdf \
	-c $OUTPUT/del_magnitud_counts.tsv  &

python $SCRIPTS/modulated_modularity_clustering.py \
	-i $DATA \
	-d $DESIGN \
	-id $ID \
	-o $OUTPUT/del_mmc_vals.tsv \
	-f $OUTPUT/del_mmc_plots.pdf &

python $SCRIPTS/partial_least_squares.py \
	-i $DATA \
	-d $DESIGN \
	-id $ID \
	-g $GROUP \
	-t $SUBSET \
	-os $OUTPUT/del_pls_scores.tsv \
	-ow $OUTPUT/del_pls_weights.tsv \
	-f $OUTPUT/del_pls_figure.pdf &

python $SCRIPTS/principal_component_analysis.py \
	-i $DATA \
	-d $DESIGN \
	-id $ID \
	-g $GROUP \
	-lo $OUTPUT/del_pca_loadings.tsv \
	-so $OUTPUT/del_pca_scores.tsv \
	-f $OUTPUT/del_pca_components.pdf &

python $SCRIPTS/random_forest.py \
	-i $DATA \
	-d $DESIGN \
	-id $ID \
	-g $GROUP \
	-o $OUTPUT/del_randforst_tab.tsv \
	-o2 $OUTPUT/del_randforst_tab.ts \
	-f $OUTPUT/del_randforst_significance.pdf &

python $SCRIPTS/standardized_eucildean_distance.py \
	-i $DATA \
	-d $DESIGN \
	-id $ID \
	-g $GROUP \
	-m $OUTPUT/del_sed_mean.tsv \
	-pw $OUTPUT/del_sed_pairwise.ts \
	-f $OUTPUT/del_sed_plot.pdf &

python $SCRIPTS/svm_classifier.py \
	-trw $DATA \
	-trd $DESIGN \
	-tew $DATA \
	-ted $DESIGN \
	-g $GROUP \
	-id $ID \
	-k linear \
	-d 3 \
	-c 1 \
	-a 0.0 \
	-b 0.0 \
	-o $OUTPUT/del_svm_table.tsv \
	-acc $OUTPUT/del_svm_accuracy.tsv &

python $SCRIPTS/threshold_based_flags.py \
	-i $DATA \
	-d $DESIGN \
	-id $ID \
	-g $GROUP \
	-o $OUTPUT/del_thrshold_flags.tsv

python $SCRIPTS/subset_data.py \
	-i $DATA \
	-d $DESIGN \
	-id $ID \
	-g $GROUP \
	-dp $SUBSET \
	-ow $OUTPUT/del_subset_data.tsv &

python $SCRIPTS/subset_data.py \
	-i $DATA \
	-d $DESIGN \
	-id $ID \
	-dp $SAMPLETOSUBSET \
	-ow $OUTPUT/del_subset_data.tsv &

python $SCRIPTS/summarize_flags.py \
	-f del_cv_flag.tsv \
	-id $ID \
	-os $OUTPUT/del_summary_flags.tsv

python $SCRIPTS/drop_flags.py \
	-i $DATA \
	-d $DESIGN \
	-f del_summary_flags.tsv \
	-id $ID \
	-fid $ID \
	-fd "flag_all_off" \
	-fft row \
	-val 1 \
	-con 2 \
	-ow $OUTPUT/del_drop_data.tsv \
	-of $OUTPUT/del_drop_flags.tsv 

python $SCRIPTS/multiple_testing_adjusment.py
	-i $OUTPUT/del_anova_table.tsv \
	-id $ID \
	-pv "p-Value of f-Value" \
	-a 0.05 \
	-on $OUTPUT/del_mta_pvals.tsv \
	-fl $OUTPUT/del_mta_flags.tsv


python $SCRIPTS/run_order_regression.py \
	-i$ $DATA \
	-d $DESIGN \
	-id $ID \
	-g $GROUP \
	-o $RUNORDER \
	-f $OUTPUT/del_ror_fig.pdf \
	-t $OUTPUT/del_ror_table.tsv
