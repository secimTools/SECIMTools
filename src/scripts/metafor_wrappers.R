library(metafor)

meta_batchCorrect <- function(data, dependent, study, treatment, factors, forest, myMethod = "FE", myMeasure = 'MD', myvtype = 'UB', toBackground = FALSE, varianceforNA = 0, commonVar = TRUE) {
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
  sd_ml[is.na(sd_ml)] <- varianceforNA

  if (commonVar) {
    print("use common variance for each batch")
    sd_common = aggregate(data[[dependent]],
                        list(batch=data[[study]]), 
                        sd)
    for (b in sd_common$batch) {
      sd_ml$x[sd_ml$batch == b] <- sd_common$x[sd_common$batch==b]
    }
  }


  ctl = length(factors)
  if (toBackground) {
    print("effect size calculated based on all controls as the background")
    num_ml_ctl = aggregate(data[[dependent]],
                 list(treatment = data[[treatment]]),
                 length)
    mean_ml_ctl = aggregate(data[[dependent]],
                      list(treatment = data[[treatment]]), 
                      mean)
    sd_ml_ctl = aggregate(data[[dependent]],
                    list(treatment = data[[treatment]]),
                    sd)
    num_ml$x[num_ml$treatment==factors[ctl]] <- num_ml_ctl$x[num_ml_ctl$treatment==factors[ctl]]
    mean_ml$x[mean_ml$treatment==factors[ctl]] <- mean_ml_ctl$x[mean_ml_ctl$treatment==factors[ctl]]
    sd_ml$x[sd_ml$treatment==factors[ctl]] <- sd_ml_ctl$x[sd_ml_ctl$treatment==factors[ctl]]
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

#res = meta_batchCorrect(data, "rawScale", "batch", "strain", ".", myMethod = "DL")
