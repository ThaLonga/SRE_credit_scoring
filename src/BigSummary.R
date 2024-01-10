#caret evaluation function for AUC, Brier score and partial Gini index
BigSummary <- function (data, lev = NULL, model = NULL) {
  print(data)
  brscore <- try(mean((data[, lev[2]] - ifelse(data$obs == lev[2], 1, 0)) ^ 2),
                 silent = TRUE)
  rocObject <- try(pROC::roc(ifelse(data$obs == lev[2], 1, 0), data[, lev[2]],
                             direction = "<", quiet = TRUE), silent = TRUE)
  #PG
  probs <- try(data[,"X1"], silent = TRUE)
  actuals <- data$obs
  print(actuals)
  sorted_indices <- order(probs, decreasing = TRUE)
  sorted_probs <- probs[sorted_indices]
  sorted_actuals <- actuals[sorted_indices]

  # Select subset with PD < 0.4
  subset_indices <- which(sorted_probs < 0.4)
  subset_probs <- sorted_probs[subset_indices]
  subset_actuals <- sorted_actuals[subset_indices]

  # Check if there are both positive and negative cases in the subset
  if (length(unique(subset_actuals)) > 1) {
    print("subset ok")
    # Calculate ROC curve for the subset
    roc_subset <- pROC::roc(subset_actuals, subset_probs,
                            direction = "<", quiet = TRUE)
    # Calculate AUC for the subset
    partial_auc <- pROC::auc(roc_subset)
    # Calculate partial Gini coefficient
    partial_gini <- 2 * partial_auc - 1
  } else {
    # Set partial Gini to NA if there are not enough cases for ROC calculation
    partial_gini <- NA
  }
    
  
  if (inherits(brscore, "try-error")) brscore <- NA
  rocAUC <- if (inherits(rocObject, "try-error")) {
    NA
  } else {
    rocObject$auc
  }
  
  return(c(AUCROC = rocAUC, Brier = brscore, partialGini = partial_gini))
}
