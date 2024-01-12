# partialGini for tidymodels
library(tidymodels, rlang)

pg_impl <- function(truth, estimate, case_weights = NULL) {

  sorted_indices <- order(estimate, decreasing = TRUE)
  sorted_probs <- estimate[sorted_indices]
  sorted_actuals <- truth[sorted_indices]

  # Select subset with PD < 0.4
  subset_indices <- which(sorted_probs < 0.4)
  subset_probs <- sorted_probs[subset_indices]
  subset_actuals <- sorted_actuals[subset_indices]

  # Check if there are both positive and negative cases in the subset
  if (length(unique(subset_actuals)) > 1) {
    # Calculate ROC curve for the subset
    roc_subset <- pROC::roc(subset_actuals, subset_probs,
                            direction = "<", quiet = TRUE)
    # Calculate AUC for the subset
    partial_auc <- pROC::auc(roc_subset)
    # Calculate partial Gini coefficient
    (2 * partial_auc - 1)
  } else return(NA)
}
    

pg_vec <- function(truth, estimate, na_rm = TRUE, case_weights = NULL, ...) {
  check_numeric_metric(truth, estimate, case_weights)
  
  if (na_rm) {
    result <- yardstick_remove_missing(truth, estimate, case_weights)
    
    truth <- result$truth
    estimate <- result$estimate
    case_weights <- result$case_weights
  } else if (yardstick_any_missing(truth, estimate, case_weights)) {
    return(NA_real_)
  }
  
  pg_impl(truth, estimate, case_weights = case_weights)
}

pg <- function(data, ...) {
  UseMethod("pg")
}

pg <- new_numeric_metric(pg, direction = "maximize")

pg.data.frame <- function(data, truth, estimate, na_rm = TRUE,...) {
  numeric_metric_summarizer(
    name = "pg",
    fn = pg_vec,
    data = data,
    truth = !! enquo(truth),
    estimate = !! enquo(estimate),
    na_rm = na_rm)
}
