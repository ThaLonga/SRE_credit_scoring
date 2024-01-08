#caret evaluation function for AUC, Brier score and partial Gini index
BigSummary <- function (data, lev = NULL, model = NULL) {
  brscore <- try(mean((data[, lev[2]] - ifelse(data$obs == lev[2], 1, 0)) ^ 2),
                 silent = TRUE)
  rocObject <- try(pROC::roc(ifelse(data$obs == lev[2], 1, 0), data[, lev[2]],
                             direction = "<", quiet = TRUE), silent = TRUE)
  #PG
  probs <- try(data[,"X2"], silent = TRUE)
  actuals <- data$obs
  sorted_indices <- order(probs, decreasing = TRUE)
  sorted_probs <- probs[sorted_indices]
  sorted_actuals <- actuals[sorted_indices]
  
  # Select subset with PD < 0.4
  subset_indices <- which(sorted_probs < 0.4)
  subset_probs <- sorted_probs[subset_indices]
  subset_actuals <- sorted_actuals[subset_indices]
  
  # Calculate ROC curve for the subset
  roc_subset <- pROC::roc(subset_actuals, subset_probs, quiet = TRUE)
  # Calculate AUC for the subset
  partial_auc <- pROC::auc(roc_subset)
  # Calculate partial Gini coefficient
  partial_gini <- 2 * partial_auc - 1
  
  if (inherits(brscore, "try-error")) brscore <- NA
  rocAUC <- if (inherits(rocObject, "try-error")) {
    NA
  } else {
    rocObject$auc
  }
  
  return(c(AUCROC = rocAUC, Brier = brscore, partialGini = partial_gini))
}