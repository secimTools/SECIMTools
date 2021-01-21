#! /bin/sh
# Copyright (C) 2020 Oleksandr Moskalenko <om@rc.ufl.edu>
# Distributed under terms of the MIT license.

SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})
${SCRIPT_DIR}/anova_fixed_with_group.sh
${SCRIPT_DIR}/bland_altman_plot_with_group.sh
${SCRIPT_DIR}/blank_feature_filtering_flags.sh
${SCRIPT_DIR}/coefficient_variation_flags_with_group.sh
${SCRIPT_DIR}/compare_flags.sh
${SCRIPT_DIR}/compound_identification.sh
${SCRIPT_DIR}/data_normalization_and_rescailing_mean.sh
${SCRIPT_DIR}/data_normalization_and_rescailing_vast.sh
${SCRIPT_DIR}/distribution_features_with_group.sh
${SCRIPT_DIR}/distribution_samples_with_group.sh
${SCRIPT_DIR}/hierarchical_clustering_heatmap.sh
${SCRIPT_DIR}/imputation.sh
${SCRIPT_DIR}/kruskal_wallis_with_group.sh
exit 0
#TODO
${SCRIPT_DIR}/lasso_enet_var_select.sh
${SCRIPT_DIR}/linear_discriminant_analysis_none.sh
${SCRIPT_DIR}/log_and_glog_transformation_glog_lambda_1000000.sh
${SCRIPT_DIR}/log_and_glog_transformation_log.sh
${SCRIPT_DIR}/magnitude_difference_flags_with_group.sh
${SCRIPT_DIR}/mahalanobis_distance.sh
${SCRIPT_DIR}/merge_flags.sh
${SCRIPT_DIR}/modify_design_file.sh
${SCRIPT_DIR}/modulated_modularity_clustering.sh
${SCRIPT_DIR}/multiple_testing_adjustment.sh
${SCRIPT_DIR}/mzrt_match.sh
${SCRIPT_DIR}/partial_least_squares_none.sh
${SCRIPT_DIR}/principal_component_analysis.sh
${SCRIPT_DIR}/random_forest.sh
${SCRIPT_DIR}/remove_selected_features_samples.sh
${SCRIPT_DIR}/retention_time_flags.sh
${SCRIPT_DIR}/run_all.sh
${SCRIPT_DIR}/run_order_regression.sh
${SCRIPT_DIR}/scatter_plot_2D_default.sh
${SCRIPT_DIR}/scatter_plot_2D_palette_color.sh
${SCRIPT_DIR}/scatter_plot_3D_default.sh
${SCRIPT_DIR}/scatter_plot_3D_palette_color.sh
${SCRIPT_DIR}/standardized_euclidean_distanc.sh
${SCRIPT_DIR}/summarize_flags.sh
${SCRIPT_DIR}/svm_classifier_none.sh
${SCRIPT_DIR}/threshold_based_flags.sh
${SCRIPT_DIR}/ttest_select_unpaired.sh
${SCRIPT_DIR}/ttest_single_group_no_group.sh
${SCRIPT_DIR}/ttest_single_group_with_group.sh
