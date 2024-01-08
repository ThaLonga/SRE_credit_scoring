# to import model templates to make the main code simpler 

#####################
# LR-R
#####################
LR_R <- function(data, n_folds = 10, seed) {
  LR_R_ctrl = trainControl(method = "cv", number = n_folds, classProbs = TRUE, summaryFunction = BigSummary)
  
  data = data  %>% 
    mutate(label = factor(label, 
                          labels = make.names(levels(label))))
  
  set.seed(seed)
  LRR_model <- train(label ~., data = train,  method = "glmnet", trControl = LR_R_ctrl, metric = metric,
                     tuneGrid = expand.grid(alpha = hyperparameters_LR_R$alpha,lambda = hyperparameters_LR_R$lambda),
                     allowParallel=TRUE)
  return(predict(LRR_model, x_test, type = "prob"))
}

source("./src/hyperparameters.R")

LR_R_preds <- LR_R(train, seed = 123)
#best tune: alpha = 0.1, lambda = 0.1
#coef(LRR_model$finalModel, LRR_model$bestTune$lambda)
partialGini(LR_R_preds, y_test)
