library(metafor)

meta_batchCorrect <- function(data, dependent, study, treatment, factors, forest, myMethod = "FE", myMeasure = 'SMD', myvtype = 'LS', toBackground = FALSE) {
  batch_set = rep(c("set1", "set2", "set3"), each = 2)
  names(batch_set) = c(1, 2, 3, 4, 5, 6)
  data$sets <- batch_set[data[[study]]]
  varianceforNA <- sd(data[[dependent]])

  num_ml = aggregate(data[[dependent]],
                     list(batch=data[[study]],
                     treatment = data[[treatment]]), 
                     length)
  mean_ml = aggregate(data[[dependent]],
                      list(batch=data[[study]], 
                      treatment = data[[treatment]]), 
                      mean)
  sd_ml = aggregate(data[[dependent]],
                      list(batch=data[[study]], 
                      treatment = data[[treatment]]), 
                      sd)
  #sd_ml[is.na(sd_ml)] <- 0.01
  sd_ml[is.na(sd_ml)] <- 0.01
  sd_ml$x[sd_ml$x==0] <- 0.01


  ctl = length(factors)
  if (toBackground) {
    print("use common variance for each batch")
    num_ml = replaceSetData(num_ml, length, dependent, ctl, data, factors)
    mean_ml = replaceSetData(mean_ml, mean, dependent, ctl, data, factors)
    sd_ml= replaceSetData(sd_ml, sd, dependent, ctl, data, factors)
  }
  else {
    print("default metafor behavior")
  }

  es_all = data.frame()
  slabs = c()

  for (i in 1:(length(factors)-1)) {
    studies = num_ml[[study]][num_ml$treatment==factors[i]]
    studies_names = paste(factors[i], "_batch_", as.character(studies), sep='')
    estimate_ml = escalc(measure= myMeasure, 
                       n1i= num_ml$x[num_ml$treatment==factors[i]],
                       n2i= num_ml$x[(num_ml$batch %in% studies) & (num_ml$treatment==factors[ctl])],
                       m1i= mean_ml$x[mean_ml$treatment==factors[i]],
                       m2i= mean_ml$x[(mean_ml$batch %in% studies) & (mean_ml$treatment==factors[ctl])],
                       sd1i= sd_ml$x[sd_ml$treatment==factors[i]],
                       sd2i= sd_ml$x[(sd_ml$batch %in% studies) & (sd_ml$treatment==factors[ctl])],
                       vtype = myvtype, 
                       slab = studies_names)
    rownames(estimate_ml) = studies
    es_all <- rbind(es_all, as.data.frame(estimate_ml))
    slabs <- c(slabs, studies_names)
  }

  meta_ml = rma(yi, vi,
                method = myMethod,
                slab = slabs, 
                data=es_all)

  #resid = residuals(meta_ml)
  #fitted = fitted(meta_ml)
  print(forest)

  if (forest != "NOFIG") {
    pdf(file = forest)
    #forest = forest(meta_ml, main = paste("Effect of ", factors[1], " in each ", study, sep=''))
    plot(meta_ml)
    dev.off()
  }


  #if (myMethod == 'DL') {
    #conf = confint(meta_ml)
    #predict(meta_ml) 
    #blup(meta_ml)
    #ranef(meta_ml)
    #cumul(meta_ml)
  #}
  res = getTestResults(meta_ml, digits = 6)
  print(meta_ml)
  return(res)
}

getTestResults <- function(model, digits = 6) {
  effect = round(model$b, digits =digits)
  se = round(model$se, digits =digits)
  zv = round(model$zval, digits =digits)
  pv = round(model$pval, digits =digits)
  ci_l = round(model$ci.lb, digits =digits)
  ci_u = round(model$ci.ub, digits =digits)
  
  meta_summary = c(effect, se, zv, pv, ci_l, ci_u)
  names(meta_summary) <- c("effct", "se", "z_value", "p_value", "ci_lb", "ci_ub")
  return(meta_summary)
}


replaceSetData <- function(x, fun, dependent, ctl, data, factors) {
  common = aggregate(data[[dependent]],
                        list(sets=data[["sets"]]), 
                        fun)
  for (b in common$sets) {
    x$x[(x$sets == b) & (x$treatment == factors[ctl])] <- common$x[common$sets == b]
  }
  return(x)
}


