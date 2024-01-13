# partialGini for tidymodels
library(tidymodels, rlang)

pg_impl <- function(truth,
                    estimate,
                    case_weights){
  pg_factor(truth = truth,
            estimate = estimate,
            case_weights = case_weights)
}

pg_ind <- function(truth, estimate, case_weights = NULL) {
  
  if (is.vector(truth)) {
    truth <- matrix(truth, ncol = 1)
  }
  
  if (is.vector(estimate)) {
    estimate <- matrix(estimate, ncol = 1)
  }
  # In the binary case:
  if (ncol(estimate) == 1 && ncol(truth) == 2) {
    estimate <- unname(estimate)
    estimate <- vec_cbind(estimate, 1 - estimate, .name_repair = "unique_quiet")
  }
  

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
  } else {return(NA)}
}
    

pg_vec <- function(truth, estimate, na_rm = TRUE, case_weights = NULL, ...) {
  abort_if_class_pred(truth)
  
  estimator <- finalize_estimator(truth, metric_class = NULL)
  check_prob_metric(truth, estimate, case_weights, estimator)
  
  if (na_rm) {
    result <- yardstick_remove_missing(truth, estimate, case_weights)
    truth <- result$truth
    estimate <- result$estimate
    case_weights <- result$case_weights
    
  } else if (yardstick_any_missing(truth, estimate, case_weights)) {
    return(NA_real_)
  }
  
  pg_impl(truth = truth, estimate = estimate, case_weights = case_weights)
  
}

pg <- function(data, ...) { #OK
  UseMethod("pg")
}

pg <- new_prob_metric(pg, direction = "maximize") # OK

pg.data.frame <- function(data, truth, ..., na_rm = TRUE, case_weights = NULL) {
  case_weights_quo <- enquo(case_weights)
  
  prob_metric_summarizer(
    name = "pg",
    fn = pg_vec,
    data = data,
    truth = !! enquo(truth),
    ...,
    na_rm = na_rm,
    case_weights = !!case_weights_quo)
}

pg_factor <- function(truth, estimate, case_weights = NULL) {
  inds <- hardhat::fct_encode_one_hot(truth)
  
  pg_ind(inds, estimate, case_weights)
}
