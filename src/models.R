# to import model templates to make the main code simpler 
#returns predictions

#####################
# LR-R
#####################
LR_R <- function(recipe, data = train, n_folds = 10, seed) {
  LR_R_ctrl = trainControl(method = "cv", number = n_folds, classProbs = TRUE, summaryFunction = BigSummary)
  
  #train = train  %>% 
    #mutate(label = factor(label, 
                          #labels = make.names(levels(label))))
  
  set.seed(seed)
  LRR_model <- train(recipe, data = data,  method = "glmnet", trControl = LR_R_ctrl, metric = metric,
                     tuneGrid = expand.grid(alpha = hyperparameters_LR_R$alpha,lambda = hyperparameters_LR_R$lambda),
                     allowParallel=TRUE)
  return(predict(LRR_model, test, type = "prob"))
}

get_splineworthy_columns <- function(X) {
  return(lapply(X, n_distinct)>6)
}

#PG: cutoff = max probability of default
partialGini <- function(preds, actuals, cutoff = 0.4) {
  
  # Sort observations by predicted probabilities
  sorted_indices <- order(preds, decreasing = TRUE)
  sorted_preds <- preds[sorted_indices]
  sorted_actuals <- actuals[sorted_indices]
  
  # Select subset with PD < 0.4
  subset_indices <- which(sorted_preds < cutoff)
  subset_preds <- sorted_preds[subset_indices]
  subset_actuals <- sorted_actuals[subset_indices]
  
  # Calculate ROC curve for the subset
  roc_subset <- pROC::roc(subset_actuals, subset_preds)
  
  # Calculate AUC for the subset
  partial_auc <- pROC::auc(roc_subset)
  
  # Calculate partial Gini coefficient
  partial_gini <- 2 * partial_auc - 1
  return(partial_gini)
}
