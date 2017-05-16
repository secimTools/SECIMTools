#!/bin/bash

echo "Running anova.py"
# Anova
python anova.py \
    --input ./test-data/test_data.tsv \
    --design ./test-data/test_design.csv \
    --ID rowID \
    --group treatment \
    --out ./test-data/test_anova_tsv.tsv \
    --fig ./test-data/test_anova_fig.pdf \
    --fig2 ./test-data/test_anova_fig2.pdf
echo "\n\n"

#------------------------------------------

echo "Running baPlot.py"
# Ba Plot
python baPlot.py \
    --input ./test-data/test_data.tsv \
    --design ./test-data/test_design.csv \
    --ID rowID \
    --group treatment \
    --ba ./test-data/test_baPlot_ba.pdf \
    --flag_dist ./test-data/test_baPlot_flag_dist.pdf \
    --flag_sample ./test-data/test_baPlot_flag_sample.tsv \
    --flag_feature ./test-datatest_baPlot_flag_feature.tsv
echo "\n\n"

#------------------------------------------

echo "Running clean.py"
# Clean
python clean.py \
    --input ./test-data/test_data.tsv \
    --design ./test-data/test_design.csv \
    --ID rowID \
    --wideOut ./test-data/test_clean_wideOut.tsv \
    --designOut ./test-data/test_clean_designOut.tsv
echo "\n\n"

#------------------------------------------

echo "Running countDigits.py"
# Count Digits
python countDigits.py \
    --input ./test-data/test_data.tsv \
    --design ./test-data/test_design.csv \
    --ID rowID \
    --html_path ./test-data/html-path/ \
    --html ./test-data/html-path/html.html \
    --flags ./test-data/html-path/flags.tsv \
    --noZip \
    --group treatment
echo "\n\n"

#------------------------------------------

echo "Running CVflag.py"
# CV Flag
python CVflag.py \
    --input ./test-data/test_data.tsv \
    --design ./test-data/test_design.csv \
    --ID rowID \
    --CVplotOutFile ./test-data/test_CVplotOutFile.pdf \
    --CVflagOutFile ./test-data/test_CVflagOutFile.pdf \
    --group treatment
echo "\n\n"

#------------------------------------------

echo "Running distribution.py"
# Distribution
python distribution.py \
    --input ./test-data/test_data.tsv \
    --design ./test-data/test_design.csv \
    --ID rowID \
    --fig ./test-data/test_distribution_fig.pdf \
    --fig2 ./test-data/test_distribution_fig2.html \
    --group treatment
echo "\n\n"

#------------------------------------------

echo "Running dropFlag.py"
# dropFlag
python dropFlag.py \
    --input ./test-data/test_data.tsv \
    --design ./test-data/test_design.csv \
    --ID rowID \
    --group treatment \
    --flags ./test-data/test_flag.tsv \
    --cutoff .5 \
    --wideOut ./test-data/test_dropFlag_wideOut.tsv \
    --designOut ./test-data/test_dropFlag_designOut.tsv \
    --row
echo "\n\n"

#------------------------------------------

# Hierarchical Cluster
#???

#------------------------------------------

echo "Running log_transform.py"
# Log Transform
python log_transform.py \
    --input ./test-data/test_data.tsv \
    --design ./test-data/test_design.csv \
    --ID rowID \
    --log log \
    --out ./test-data/test_log_transform_out.tsv
echo "\n\n"

#------------------------------------------

echo "Running mean_standardize.py"
# Mean Standardize
python mean_standardize.py \
    --input ./test-data/test_data.tsv \
    --design ./test-data/test_design.csv \
    --ID rowID \
    --group treatment \
    --std MEAN \
    --out ./test-data/test_mean_standardize_out.tsv
echo "\n\n"

#------------------------------------------

echo "Running mergeFlags.py"
# Merge Flags
python mergeFlags.py \
        --input ./test-data/test_flag.tsv \
        --output ./test-data/test_mergeFlags_output.tsv
echo "\n\n"

#------------------------------------------

echo "Running onOff.py"
# On/Off
python onOff.py \
    --input ./test-data/test_data.tsv \
    --design ./test-data/test_design.csv \
    --ID rowID \
    --output ./test-data/test_onOff_output.tsv \
    --group treatment \
    --cutoff 30000
echo "\n\n"

#------------------------------------------

echo "Running pypca.py"
# Pypca
python pypca.py \
    --input ./test-data/test_data.tsv \
    --design ./test-data/test_design.csv \
    --ID rowID \
    --load_out ./test-data/test_pypca_load_out.tsv \
    --score_out ./test-data/test_pypca_score_out.tsv
echo "\n\n"

#------------------------------------------


echo "Running RandomForest.py"
# Random Forest
python RandomForest.py \
    --input ./test-data/test_data.tsv \
    --design ./test-data/test_design.csv \
    --ID rowID \
    --group treatment \
    --num 1000 \
    --out ./test-data/test_RandomForest_out.tsv \
    --out2 ./test-data/test_RandomForest_out2.tsv
echo "\n\n"

#------------------------------------------

echo "Running RTflag.py"
# RT Flag
python RTflag.py \
    --input ./test-data/test_data.tsv \
    --design ./test-data/test_design.csv \
    --ID rowID \
    --RTplotOutFile ./test-data/test_RTflag_RTplotOutFile.tsv \
    --RTflagOutFile ./test-data/test_RTflag_RTplotOutFile.png \
    --minutes .2
echo "\n\n"

#------------------------------------------

echo "Running runOrderRegression.py"
# Run Order
python runOrderRegression.py \
    --input ./test-data/test_data.tsv \
    --design ./test-data/test_design.csv \
    --ID rowID \
    --group treatment \
    --order runOrder \
    --fig ./test-data/test_runOrderRegression_fig.pdf \
    --table ./test-data/test_runOrderRegression_table.pdf
echo "\n\n"

#------------------------------------------

echo "Running scatterPlot3D.py"
 Scatter Plot 3D
python scatterPlot3D.py \
    --input ./test-data/test_data.tsv \
    --design ./test-data/test_design.csv \
    --ID rowID \
    --group treatment \
    --fig ./test-data/test_scatterPlot3D.png \
    --xaxis PC1 \
    --yaxis PC2 \
    --zaxis PC3
echo "\n\n"

#------------------------------------------

echo "Running scatterPlot.py"
# Scatter Plot
python scatterPlot.py \
    --input ./test-data/test_data.tsv \
    --design ./test-data/test_design.csv \
    --ID rowID \
    --group treatment \
    --fig ./test-data/test_scatterPlot.png \
    --xaxis PC1 \
    --yaxis PC2
echo "\n\n"

#------------------------------------------

echo "Running standardizedEuclideanDistance.py"
# Standard Euc Distance
python standardizedEuclideanDistance.py \
    --input ./test-data/test_data.tsv \
    --design ./test-data/test_design.csv \
    --ID rowID \
    --group treatment \
    --SEDplotOutFile ./test-data/test_standardizedEuclideanDistance_SEDplotOutFile.pdf \
    --SEDtoCenter ./test-data/test_standardizedEuclideanDistance_SEDtoCenter.tsv \
    --SEDpairwise ./test-data/test_standardizedEuclideanDistance_SEDpairwise.tsv
echo "\n\n"

#------------------------------------------

echo "Running SVM_classifier.py"
# SVM
python SVM_classifier.py \
    --train_wide ./test-data/test_data.tsv \
    --train_design ./test-data/test_design.csv \
    --test_wide ./test-data/test_data.tsv \
    --test_design ./test-data/test_design.csv \
    --class_column_name treatment \
    --ID rowID \
    --kernel linear \
    --degree 3 \
    --C 1 \
    --a 0.0 \
    --b 0.0 \
    --outfile ./test-data/test_SVM_classifier_outfile.tsv \
    --accuracy_on_training ./test-data/test_SVM_classifier_accuracy_on_training.tsv
echo "\n\n"

