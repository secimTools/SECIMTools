library(metafor)

meta_batchCorrect <- function(data, dependent, study, treatment, factors, forest, myMethod = "FE", myMeasure = 'MD') {
  num_ml = aggregate(data[[dependent]],
                     list(study=data[[study]], 
                     treatment = data[[treatment]]), 
                     length)
  mean_ml = aggregate(data[[dependent]],
                      list(study=data[[study]], 
                      treatment = data[[treatment]]), 
                      mean)
  sd_ml = aggregate(data[[dependent]],
                    list(study=data[[study]], 
                    treatment = data[[treatment]]), 
                    sd)
  
  estimate_ml = escalc(measure=myMeasure, 
                     n1i= num_ml$x[num_ml$treatment==factors[1]],
                     n2i= num_ml$x[num_ml$treatment==factors[2]],
                     m1i= mean_ml$x[mean_ml$treatment==factors[1]],
                     m2i= mean_ml$x[mean_ml$treatment==factors[2]],
                     sd1i= sd_ml$x[sd_ml$treatment==factors[1]],
                     sd2i= sd_ml$x[sd_ml$treatment==factors[2]], 
                     append=T)

  rownames(estimate_ml) = num_ml[[study]][num_ml$treatment==factors[1]]

  meta_ml = rma(yi, vi,
                measure = myMeasure,
                method = myMethod, 
                data=estimate_ml)

  #resid = residuals(meta_ml)
  #fitted = fitted(meta_ml)
  #pdf(file = forest)
  #forest = forest(meta_ml)
  #dev.off()
  #if (myMethod == 'DL') {
    #conf = confint(meta_ml)
    #predict(meta_ml) 
    #blup(meta_ml)
    #ranef(meta_ml)
    #cumul(meta_ml)
  #}
  res = getTestResults(meta_ml, digits = 4)
  print(meta_ml)
  return(res)
}

getTestResults <- function(model, digits = 4) {
  se = round(model$se, digits =digits)
  zv = round(model$zval, digits =digits)
  pv = round(model$pval, digits =digits)
  ci_l = round(model$ci.lb, digits =digits)
  ci_u = round(model$ci.ub, digits =digits)
  
  meta_summary = c(se, zv, pv, ci_l, ci_u)
  names(meta_summary) <- c("se", "z_value", "p_value", "ci_lb", "ci_ub")
  return(meta_summary)
}

#res = meta_batchCorrect(data, "rawScale", "batch", "strain", ".", myMethod = "DL")
