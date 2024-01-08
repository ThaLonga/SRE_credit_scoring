# to import model templates to make the main code simpler 
#returns predictions

#####################
# LR-R
#####################
LR_R <- function(train, test, n_folds = 10, seed) {
  LR_R_ctrl = trainControl(method = "cv", number = n_folds, classProbs = TRUE, summaryFunction = BigSummary)
  
  train = train  %>% 
    mutate(label = factor(label, 
                          labels = make.names(levels(label))))
  
  set.seed(seed)
  LRR_model <- train(label ~., data = train,  method = "glmnet", trControl = LR_R_ctrl, metric = metric,
                     tuneGrid = expand.grid(alpha = hyperparameters_LR_R$alpha,lambda = hyperparameters_LR_R$lambda),
                     allowParallel=TRUE)
  return(predict(LRR_model, test, type = "prob"))
}

