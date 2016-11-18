# 2016.08.05 ask
# The goal is to implement elasticnet for the available data so that we 
# can use it for variable selection techniquest and prediction.
# This analysis can be ported to Python later.

#rm(list=ls(all=TRUE))

# In this piece of code we are looking for LODGR filtered data i.e. we used LOD flag for QC samples
# and LOD flags for each group (4 groups for our case) and in this case and filtered metabolited
# provided it was flagged in BOTH QC LOD and in ALL Group LOD-s.

# We are doign it for BOTH AREA and HEIGHT!!!

# Setting the correct working directory.
# NOTE!!! -> SHARE can be linked differently on different computers.
# setwd("~/SHARE/SHARE/McIntyre_Lab/Alexander/lasso_en")

#setwd("/home/mthoburn/Desktop/work/galaxy/tools/GalaxyTools")

# setwd("E:/lasso")
# Extra check

#getwd()

# Reading the design file into R

#Design_file <- read.table("test-data/real_design.tsv", sep= "\t", fill = TRUE, header = TRUE)

# Checking the dimension of what we just read.

#dim(Design_file)

# Reading varsel datasets into R.

#Dataset_varsel <- read.table("test-data/alex_data.tsv", sep= "\t", fill = TRUE, header = TRUE, check.names = FALSE )

# Checking the dimension of what we just read.

#dim(Dataset_varsel)


# Package for permutations

#library(caTools)

# Defingin pairs of comparisons we gonna do LASSO/ElasticNet for.
# WARNING!!! It has to be the same for POS and NEG for loops to run successfully.
# Therefore we choose compinations only from the ones that are in both files.
# NOTE: If there is one desing file those things should be adjusted accordingly.
# We need to delete single occurances otherwise LASSO/ElasticNet not gonna run.
# Getting index(es) to delete

#group_levels_to_delete_indes <- which( summary(Design_file$group) %in% c(1,2) )

# Getting actual variables to delete

#group_levels_to_delete <- names(summary(Design_file$group))[ group_levels_to_delete_indes ]

# Actual deletion procedure. We are keeping only the ones that donot contain levels we had intent to delete.

#pairs_comparison <- combs( unique(as.character(Design_file$group))[!(unique(as.character(Design_file$group)) %in% group_levels_to_delete)]   , 2) 
#pairs_comparison

# Converting to data frame
# pairs_comparison <- data.frame(pairs_comparison)
# Getting the length

#pairs_length <- length(pairs_comparison[,1])
#pairs_length

#FUNCTION START HERE

# Here we specify the parameter for the 
# alpha=1 is the lasso penalty, and alpha=0 the ridge penalty
# The middle is 

#alpha <- 0.5



# Debugging piece
# pair <-2
# pairs_comparison[pair,]
# ion <- "NEG"
# ion
# area_height <- "PA"
# area_height
lassoEN <- function(Dataset_varsel,Design_file,pairs_comparison,pairs_length,alpha,plots){
# Looping over possibel comparisons i.e. over the rows of pairs_length i.e. over the rows 1 .... pairs_length.
  for ( pair in c(1:pairs_length) )
  {
    
    # Selecting subdataset with only the levels of interest. 
    # Dataset with only groups on interest i.e. pairs_comparison[pair,]
    assign( paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_only",  sep =''),  
            Dataset_varsel[  as.character(Dataset_varsel$group) %in% as.character(pairs_comparison[pair,])  , ] )
    
    # Extra check of what we did.
    # eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_only",  sep ='')))[,2]
    
    
    # Creating design matrix with only the levels of interest.
    # Dataset with only groups on interest i.e. pairs_comparison[pair,]
    assign( paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_only_matrix",  sep =''),  
            model.matrix( group~. ,   data = eval(as.name( paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_only",  sep ='') ))[
              !(names(eval(as.name( paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_only",  sep ='') )))  %in% "sampleID")    ]   )[,-1]  )
    # Check of what we just did
    # eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_only_matrix",  sep ='')))
    
    # Response 0-1 coding
    # LOGGED Version
    # Getting indexes where 1-s should be.
    index_one <- which( eval(as.name( paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_only",  sep ='') ))$group == pairs_comparison[pair,1]  )
    
    # Creating response vector 
    response_vector <- rep(0, dim( eval(as.name( paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_only",  sep ='') )) )[1] ) 
    # Filling 1-s in response vector where necessary.
    response_vector[index_one] <- 1
    response_vector
    
    # Creating response vector with appropriate naming from our temporal response_vector
    assign( paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_response",  sep =''),  response_vector )
    
    # Extra check of what we did.
    # eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_response",  sep ='') ))
    
  }
  ls()
  
  
  # 2016.08.05.
  # Actual elasticnet piece
  
  # library(glmnet) can actually fit something more fancy. Int can do ELASTIC net with both LASSO and RIDGE Regression
  # The extreme cases for Elastic NEt are LASSO and Ridge Regression.
  # Here we are performing cross-validation and picking variables based on Cross Validation statistics.
  library(glmnet)
  
  
  # Open the file thread down to write the report.
  pdf( paste(plots,  sep =''), height = 8, width = 16 )
  
  # Looping over possibel comparisons i.e. over the rows of pairs_length i.e. over the rows 1 .... pairs_length.
  for ( pair in c(1:pairs_length) )
  {
    
        # Fitting elasticnet without cross validation. Just single run.
        assign( paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_glmnet_elasticnet",  sep =''),  
                glmnet(x = eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_only_matrix",  sep ='') )),
                          y = eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_response",  sep ='') )), alpha= alpha, family='binomial')  )
        # Extra check of what we did.
        # eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_glmnet_elasticnet",  sep ='')))
    
        # Fitting elasticnet WITH cross validation. 
        assign( paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_cv_glmnet_elasticnet",  sep =''),  
                cv.glmnet(x = eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_only_matrix",  sep ='') )),
                       y = eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_response",  sep ='') )), alpha = alpha, family='binomial', 
                       nfolds = length(  eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_response",  sep ='') ))  )  ) ) 
        # Extra check of what we did.
        # eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_cv_glmnet_elasticnet",  sep ='')))
        
        
        # Plotting the results.
        # In the separate files.
        # elasticnet without cross validation. Just single run.
  
        plot( eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_glmnet_elasticnet",  sep =''))) )
        title(main = paste("Coefficients for ", pairs_comparison[pair,1]," vs ", pairs_comparison[pair,2]," Based on Elastic Net Penalty ( alpha = ", alpha, " )",  sep =''), line = 3 )
  
        # elasticnet WITH cross validation. Just single run.
        plot( eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_cv_glmnet_elasticnet",  sep ='')))  )
        title(main = paste("Cross-Varidation Results for ", pairs_comparison[pair,1]," vs ", pairs_comparison[pair,2]," Based on Elastic Net Penalty ( alpha = ", alpha, " )",  sep =''), line = 3 )
  
        # Getting the best lambda
        assign( paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_cv_glmnet_elasticnet_best_lambda",  sep =''),  
                eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_cv_glmnet_elasticnet",  sep ='')))$lambda.min )
        # Extra check      
        # eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_cv_glmnet_elasticnet_best_lambda",  sep ='')))
  
  
        # Getting the subset index that corresponds to the lambda.min via cross-validation.
        assign( paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_cv_glmnet_elasticnet_lambda_min_index",  sep =''),  
                which ( eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_cv_glmnet_elasticnet",  sep ='')))$lambda ==  
                          eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_cv_glmnet_elasticnet",  sep ='')))$lambda.min )  )
        # Extra check      
        # eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_cv_glmnet_elasticnet_lambda_min_index",  sep ='')))
  
        
        # Pullling all coefficients (zero and nonzero) into a column of the data frame
        # For the first one we just create a vector of all coefficients.
        if (pair == 1)
        {
          
          # PUlling coefficients here.
          elasticnet_coefficients <- data.frame( coef(eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_cv_glmnet_elasticnet",  sep ='')))$glmnet.fit)[,
                eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_cv_glmnet_elasticnet_lambda_min_index",  sep =''))) ] )
          elasticnet_coefficients <- data.frame( rownames(elasticnet_coefficients), elasticnet_coefficients )
          names(elasticnet_coefficients) <- c( "rowID", paste(pairs_comparison[pair,1],"_vs_", pairs_comparison[pair,2],  sep ='') )
          rownames(elasticnet_coefficients) <- NULL
          
  
          # PUlling flags here.
          elasticnet_flags <- data.frame( as.integer(coef(eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_cv_glmnet_elasticnet",  sep ='')))$glmnet.fit)[,
                eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_cv_glmnet_elasticnet_lambda_min_index",  sep =''))) ] != 0 ) )
          elasticnet_flags <- data.frame( elasticnet_coefficients$rowID , elasticnet_flags  )
          names(elasticnet_flags) <- c( "rowID", paste(pairs_comparison[pair,1],"_vs_", pairs_comparison[pair,2],"_selection_flag_on",  sep ='') )
          
          
                  
        }  
        if (pair > 1)
        {
          # PUlling coefficients here.
          elasticnet_coefficients <- data.frame( elasticnet_coefficients,  coef(eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_cv_glmnet_elasticnet",  sep ='')))$glmnet.fit)[,
               eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_cv_glmnet_elasticnet_lambda_min_index",  sep =''))) ], check.names = FALSE )
          # Getting the current number of columns so that we can rename the last one.
          num_columns <- dim(elasticnet_coefficients)[2]
          names(elasticnet_coefficients)[num_columns] <- paste(pairs_comparison[pair,1],"_vs_", pairs_comparison[pair,2],  sep ='')
          
  
          # PUlling flags here.
          elasticnet_flags <- data.frame( elasticnet_flags,  as.integer(coef(eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_cv_glmnet_elasticnet",  sep ='')))$glmnet.fit)[,
               eval(as.name(paste("Dataset_varsel_", pairs_comparison[pair,1],"_", pairs_comparison[pair,2],"_cv_glmnet_elasticnet_lambda_min_index",  sep =''))) ] != 0) , check.names = FALSE )
          # Getting the current number of columns so that we can rename the last one.
          num_columns <- dim(elasticnet_flags)[2]
          names(elasticnet_flags)[num_columns] <- paste(pairs_comparison[pair,1],"_vs_", pairs_comparison[pair,2],"_selection_flag_on",  sep ='') 
          
          
  
        }  
  
  }
  dev.off()
  returnList <- list(elasticnet_coefficients,elasticnet_flags)
  return(returnList)
  # Writing coefficients to the table.
  # CSV
  write.table( elasticnet_coefficients, file = "elasticnet_coefficients.csv", append = FALSE, quote = FALSE, sep = ",", row.names = FALSE, col.names = TRUE )
  # TSV
  write.table( elasticnet_coefficients, file = "elasticnet_coefficients.tsv", append = FALSE, quote = FALSE, sep = "\t", row.names = FALSE, col.names = TRUE )
  
  
  # Writing flags to the table.
  # CSV
  write.table( elasticnet_flags, file = "elasticnet_flags.csv", append = FALSE, quote = FALSE, sep = ",", row.names = FALSE, col.names = TRUE )
  # TSV
  write.table( elasticnet_flags, file = "elasticnet_flags.tsv", append = FALSE, quote = FALSE, sep = "\t", row.names = FALSE, col.names = TRUE )
}

