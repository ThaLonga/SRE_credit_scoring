# main
if (!require("pacman")) install.packages("pacman") ; require("pacman")
p_load(glmnet, glmnetUtils, mgcv, tidyverse, xgboost, DiagrammeR, stringr, tictoc, parallel, pROC, earth, Matrix, pre, caret, parsnip, ggplot2, recipes, rsample, workflows, healthyR.ai)
#conventions:
# target variable = label
# indepentent variables = all others

#levels should be X1 and X2
#evaluation function for caret tuning from pre package
source("./src/models.R")
source("./src/hyperparameters.R")
source("./src/BigSummary.R")
source("./src/data_loader.R")
datasets <- load_data()
cl <- makeCluster(detectCores()-1)
#(AUCROC, Brier, partialGini)
metric = "AUCROC"
nr_repeats = 5
outerfolds = 2
innerfolds = 5
dataset_vector = c("GC", "AC", "GMSC", "TH02")

ctrl <- trainControl(method = "cv", number = innerfolds, classProbs = TRUE, summaryFunction = BigSummary)




# create empty dataframe metric_results with columns: (dataset, repeat, fold, algorithm, metric)	
metric_results <- data.frame(
  dataset = character(),
  nr_fold = integer(),
  algorithm = character(),
  metric = double(),
  stringsAsFactors = FALSE
)
#metric_results[nrow(metric_results) + 1,] = list(dataset, 2, 1, "test", 40.485)



dataset_counter = 1

for(dataset in datasets) {
  
  #formulate recipes
  TREE_recipe <- recipe(label ~., data = dataset) %>%
    step_impute_mean(all_numeric_predictors()) %>%
    step_impute_mode(all_string_predictors()) %>%
    step_impute_mode(all_factor_predictors())
  
  original_numeric_predictors <- names(dataset)[sapply(dataset, is.numeric)]
  LINEAR_recipe <- TREE_recipe %>%
    step_hai_winsorized_truncate(all_numeric_predictors(), fraction = 0.025) %>%
    step_select(-original_numeric_predictors)%>%
    step_normalize(all_numeric_predictors()) %>%
    step_dummy(all_string_predictors()) %>%
    step_dummy(all_factor_predictors())
  
  
  if(dataset_counter==3) {nr_repeats <- 3}
  else {nr_repeats <- 5}
  
  set.seed(123)
  # create 5x2 folds
  folds <- vfold_cv(dataset, v = outerfolds, repeats = nr_repeats, strata = NULL)
  for(i in 1:nrow(folds)) {
    cat("Fold", i, "/ 10 \n")
    train <- analysis(folds$splits[[i]])
    test <- assessment(folds$splits[[i]])

    innerseed <- i
    
    #####
    # LRR
    #####
    
    LRR_model <- train(LINEAR_recipe, data = train,  method = "glmnet", trControl = ctrl, metric = metric,
                       tuneGrid = expand.grid(alpha = hyperparameters_LR_R$alpha,lambda = hyperparameters_LR_R$lambda),
                       allowParallel=TRUE)
    LRR_preds <- predict(LRR_model, test, type = 'probs')
    print("pred ok")
    LRR_preds$label <- test$label
    g <- roc(label ~ X1, data = LRR_preds, direction = "<")
    AUC <- g$auc
    metric_results[nrow(metric_results) + 1,] = list(dataset_vector[dataset_counter], i, "LRR", AUC)
    print(AUC)
    
    #####
    # GAM
    #####
    
    
  }
  dataset_counter <- dataset_counter + 1
}
write.csv(metric_results, file = "./results/AUCROC_results.csv")

stopCluster(cl)


set.seed(123)
train_indices <- sample(1:nrow(x), 0.8 * nrow(x))
x_train <- (x[train_indices, ])
x_test <- (x[-train_indices, ])
y_train <- (y[train_indices, ])
y_test <- (y[-train_indices, ])