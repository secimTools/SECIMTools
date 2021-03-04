# The piece of code below was written to be wrapped into Python.
# The code below uses R library "glmnet" written specifically for LASSO and Elastic Net.


# The function below is used from python via interface provided by the rpy2 package.
lassoEN <- function(Dataset_varsel, Design_file, rowID_name, correct_list_of_column_names, pairs_comparison, pairs_length, alpha, plots) {

  # Looping over possible comparisons i.e. over the rows of pairs_length i.e. over the rows 1 .... pairs_length.
  for (pair in c(1:pairs_length))
  {
    # Selecting subset of data with only the levels of interest.
    # Dataset with only groups on interest i.e. pairs_comparison[pair,]
    assign(
      paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_only", sep = ""),
      Dataset_varsel[as.character(Dataset_varsel$group) %in% as.character(pairs_comparison[pair, ]), ]
    )
    # Creating design matrix with only the levels of interest.
    # Dataset with only groups on interest i.e. pairs_comparison[pair,]
    # Putting dataset into a temporary one to fix the names if necessary.
    evaluated_data <- eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_only", sep = "")))
    evaluated_names <- names(evaluated_data)
    mmatrix <- model.matrix( group~. , data = evaluated_data[!(evaluated_names %in% "sampleID")])
    current_dataset <- mmatrix[, -1]
    # Updating the names with the ones we passed to be sure we did not have "X_" instead of "_".
    colnames(current_dataset) <- correct_list_of_column_names[-length(correct_list_of_column_names)]
    assign(paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_only_matrix", sep = ""), current_dataset)

    # Response 0-1 coding
    # Getting indexes where 1-s should be.
    index_one <- which(eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_only", sep = "")))$group == pairs_comparison[pair, 1])

    # Creating response vector.
    response_vector <- rep(0, dim(eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_only", sep = ""))))[1])
    # Filling 1-s in response vector where necessary.
    response_vector[index_one] <- 1
    response_vector

    # Creating response vector with appropriate naming from our temporal response_vector
    assign(paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_response", sep = ""), response_vector)
  }
  ls()

  # Actual LASSO/ElasticNet piece.

  # library(glmnet) can use any penalty in the range from 0 to 1 moving from LASSO to ridge regression penality.
  # Here we are performing cross-validation and picking the final set of variables based on the cross validation.
  library(glmnet)

  # Open the file thread down to write the report.
  pdf(paste(plots, sep = ""), height = 8, width = 16)

  # Looping over possible comparisons i.e. over the rows of pairs_length i.e. over the rows 1 .... pairs_length.
  for (pair in c(1:pairs_length))
  {
    # Fitting elasticnet without cross validation. Just single run.
    assign(
      paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_glmnet_elasticnet", sep = ""),
      glmnet(
        x = eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_only_matrix", sep = ""))),
        y = eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_response", sep = ""))), alpha = alpha, family = "binomial"
      )
    )
    # Extra check of what we did.
    # eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_glmnet_elasticnet",  sep ='')))

    # Fitting Elastic Net WITH cross validation.
    assign(
      paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_cv_glmnet_elasticnet", sep = ""),
      cv.glmnet(
        x = eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_only_matrix", sep = ""))),
        y = eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_response", sep = ""))), alpha = alpha, family = "binomial",
        nfolds = length(eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_response", sep = ""))))
      )
    )

    # Plotting the results.
    # In the separate files.
    # elasticnet without cross validation. Just single run.

    plot(eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_glmnet_elasticnet", sep = ""))))
    title(main = paste("Coefficients for ", pairs_comparison[pair, 1], " vs ", pairs_comparison[pair, 2], " Based on Elastic Net Penalty ( alpha = ", alpha, " )", sep = ""), line = 3)

    # Elasticet WITH cross validation. Just single run.
    plot(eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_cv_glmnet_elasticnet", sep = ""))))
    title(main = paste("Cross-Varidation Results for ", pairs_comparison[pair, 1], " vs ", pairs_comparison[pair, 2], " Based on Elastic Net Penalty ( alpha = ", alpha, " )", sep = ""), line = 3)

    # Getting the best lambda
    assign(
      paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_cv_glmnet_elasticnet_best_lambda", sep = ""),
      eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_cv_glmnet_elasticnet", sep = "")))$lambda.min
    )

    # Getting the subset index that corresponds to the lambda.min via cross-validation.
    assign(
      paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_cv_glmnet_elasticnet_lambda_min_index", sep = ""),
      which(eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_cv_glmnet_elasticnet", sep = "")))$lambda ==
        eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_cv_glmnet_elasticnet", sep = "")))$lambda.min)
    )

    # Pulling all coefficients (zero and nonzero) into a column of the data frame
    # For the first one we just create a vector of all coefficients.
    if (pair == 1) {
      # Pulling coefficients here.
      elasticnet_coefficients <- data.frame(coef(eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_cv_glmnet_elasticnet", sep = "")))$glmnet.fit)[
        ,
        eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_cv_glmnet_elasticnet_lambda_min_index", sep = "")))
      ])
      elasticnet_coefficients <- data.frame(rownames(elasticnet_coefficients), elasticnet_coefficients)
      names(elasticnet_coefficients) <- c(rowID_name, paste(pairs_comparison[pair, 1], "_vs_", pairs_comparison[pair, 2], sep = ""))
      rownames(elasticnet_coefficients) <- NULL

      # Pulling flags here.
      elasticnet_flags <- data.frame(as.integer(coef(eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_cv_glmnet_elasticnet", sep = "")))$glmnet.fit)[
        ,
        eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_cv_glmnet_elasticnet_lambda_min_index", sep = "")))
      ] != 0))
      elasticnet_flags <- data.frame(elasticnet_coefficients[, 1], elasticnet_flags)
      names(elasticnet_flags) <- c(rowID_name, paste(pairs_comparison[pair, 1], "_vs_", pairs_comparison[pair, 2], "_selection_flag_on", sep = ""))
    }
    if (pair > 1) {
      # Pull coefficients
      elasticnet_coefficients <- data.frame(elasticnet_coefficients, coef(eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_cv_glmnet_elasticnet", sep = "")))$glmnet.fit)[
        ,
        eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_cv_glmnet_elasticnet_lambda_min_index", sep = "")))
      ], check.names = FALSE)
      # Getting the current number of columns so that we can rename the last one.
      num_columns <- dim(elasticnet_coefficients)[2]
      names(elasticnet_coefficients)[num_columns] <- paste(pairs_comparison[pair, 1], "_vs_", pairs_comparison[pair, 2], sep = "")

      # Pull flags
      elasticnet_flags <- data.frame(elasticnet_flags, as.integer(coef(eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_cv_glmnet_elasticnet", sep = "")))$glmnet.fit)[
        ,
        eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair, 1], "_", pairs_comparison[pair, 2], "_cv_glmnet_elasticnet_lambda_min_index", sep = "")))
      ] != 0), check.names = FALSE)
      # Getting the current number of columns so that we can rename the last one.
      num_columns <- dim(elasticnet_flags)[2]
      names(elasticnet_flags)[num_columns] <- paste(pairs_comparison[pair, 1], "_vs_", pairs_comparison[pair, 2], "_selection_flag_on", sep = "")
    }
  }
  dev.off()
  returnList <- list(elasticnet_coefficients[-1, ], elasticnet_flags[-1, ])
  return(returnList)
  # Write coefficients to output files
  write.table(elasticnet_coefficients[-1, ], file = "elasticnet_coefficients.csv", append = FALSE, quote = FALSE, sep = ",", row.names = FALSE, col.names = TRUE)
  write.table(elasticnet_coefficients[-1, ], file = "elasticnet_coefficients.tsv", append = FALSE, quote = FALSE, sep = "\t", row.names = FALSE, col.names = TRUE)
  # Write flags to output files
  write.table(elasticnet_flags[-1, ], file = "elasticnet_flags.csv", append = FALSE, quote = FALSE, sep = ",", row.names = FALSE, col.names = TRUE)
  write.table(elasticnet_flags[-1, ], file = "elasticnet_flags.tsv", append = FALSE, quote = FALSE, sep = "\t", row.names = FALSE, col.names = TRUE)
}
