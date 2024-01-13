# main
if (!require("pacman")) install.packages("pacman") ; require("pacman")
p_load(glmnet, glmnetUtils, mgcv, tidyverse, xgboost, DiagrammeR, stringr, tictoc, doParallel, pROC, earth, Matrix, pre, caret, parsnip, ggplot2, recipes, rsample, workflows, healthyR.ai, rlang, yardstick, bonsai, lightgbm)
#conventions:
# target variable = label
# indepentent variables = all others

#levels should be X1 and X2
#evaluation function for caret tuning from pre package
source("./src/models.R")
source("./src/partialGini_yardstick.R")
source("./src/hyperparameters.R")
source("./src/BigSummary.R")
source("./src/data_loader.R")
datasets <- load_data()
cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)

#(AUCROC, Brier, partialGini)
metric = "AUCROC"
nr_repeats = 5
outerfolds = 2
innerfolds = 3
dataset_vector = c("GC", "AC", "GMSC", "TH02")

ctrl <- trainControl(method = "cv", number = innerfolds, classProbs = TRUE, summaryFunction = BigSummary, search = "grid")
metrics = metric_set(roc_auc, brier_class)

# create empty dataframe metric_results with columns: (dataset, repeat, fold, algorithm, metric)	
metric_results <- data.frame(
  dataset = character(),
  nr_fold = integer(),
  algorithm = character(),
  metric = double(),
  stringsAsFactors = FALSE
)
AUC_results <- metric_results
Brier_results <- metric_results
PG_results <- metric_results

dataset_counter = 1


for(dataset in datasets) {
  
  
  # for GMSC only 3 repeats because large dataset
  if(dataset_counter==3) {nr_repeats <- 3} else {nr_repeats <- 5}
  
  set.seed(123)
  # create 5x2 folds
  folds <- vfold_cv(dataset, v = outerfolds, repeats = nr_repeats, strata = NULL)
  for(i in 1:nrow(folds)) {
    cat("Fold", i, "/ 10 \n")
    train <- analysis(folds$splits[[i]])
    test <- assessment(folds$splits[[i]])
    
    # formulate recipes
    # for tree-based algorithms
    MINIMAL_recipe <- recipe(label ~., data = train) %>%
      step_impute_mean(all_numeric_predictors()) %>%
      step_impute_mode(all_string_predictors()) %>%
      step_impute_mode(all_factor_predictors()) %>%
      step_zv(all_predictors())
    
    # for tree-based that require dummies
    TREE_recipe <- MINIMAL_recipe %>% step_dummy(all_string_predictors()) %>%
      step_dummy(all_factor_predictors()) %>%
      step_zv(all_predictors())
    
    
    # for algorithms using linear terms (LRR, gam, rule ensembles)
    LINEAR_recipe <- MINIMAL_recipe %>%
      step_hai_winsorized_truncate(all_numeric_predictors(), fraction = 0.025) %>%
      step_rm(!contains("winsorized") & where(is.numeric)) %>%
      step_normalize(all_numeric_predictors()) %>%
      step_dummy(all_string_predictors()) %>%
      step_dummy(all_factor_predictors()) %>%
      step_zv(all_predictors())
    
    

    innerseed <- i
    
    #####
    # LRR
    #####
    
    set.seed(i)
    inner_folds <- train %>% vfold_cv(v=5)
    
    
    LRR_model <- 
      parsnip::logistic_reg(
        mode = "classification",
        mixture = tune(),
        penalty = tune()
      ) %>%
      set_engine("glmnet")
    
    LRR_wf <- workflow() %>%
      add_recipe(LINEAR_recipe) %>%
      add_model(LRR_model)
    
    LRR_tuned <- tune::tune_grid(
      object = LRR_wf,
      resamples = inner_folds, #same folds as xgboost
      grid = hyperparameters_LR_R_tidy, #same setting as xgboost
      metrics = metrics,
      control = tune::control_grid(verbose = TRUE)
    )
    
    LRR_tuned %>%
      tune::show_best(metric = "brier_class") %>%
      knitr::kable()
    
    best_booster_auc <- LRR_tuned %>% select_best("roc_auc")
    final_LRR_wf_auc <- LRR_wf %>% finalize_workflow(best_booster_auc)
    final_LRR_fit_auc <- final_LRR_wf_auc %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    best_booster_brier <- LRR_tuned %>% select_best("brier_class")
    final_LRR_wf_brier <- LRR_wf %>% finalize_workflow(best_booster_brier)
    final_LRR_fit_brier <- final_LRR_wf_brier %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    
    auc <- final_LRR_fit_auc %>%
      collect_metrics() %>%
      filter(.metric == "roc_auc") %>%
      pull(.estimate)
    brier <- final_LRR_fit_brier %>%
      collect_metrics() %>%
      filter(.metric == "brier_class") %>%
      pull(.estimate)
    
    
    roc_auc_value <- final_LRR_fit %>%
      collect_metrics() %>%
      filter(.metric == "roc_auc") %>%
      pull(.estimate)
    
    
    final_LRR_fit %>%
      collect_predictions() %>% 
      roc_curve(label, .pred_X0) %>% 
      autoplot()
    
    
    #####
    # GAM
    #####
    
    #preprocessing as no hyperparameters to be tuned
    train_bake <- LINEAR_recipe %>% prep(train) %>% bake(train)
    test_bake <- LINEAR_recipe %>% prep(train) %>% bake(test)
    train_bake_x <- train_bake %>% dplyr::select(-label)
    
    smooth_vars = colnames(train_bake_x)[get_splineworthy_columns(train_bake_x)]
    formula <- as.formula(
      stringr::str_sub(paste("label ~", 
                             paste(ifelse(names(train_bake_x) %in% smooth_vars, "s(", ""),
                                   names(train_bake_x),
                                   ifelse(names(train_bake_x) %in% smooth_vars, ")",""),
                                   collapse = " + ")
      ), 0, -1)
    )
    
    GAM_model <- gam(formula, family = "binomial", data = train_bake)
    GAM_preds <- data.frame(X1 = as.vector(predict(GAM_model, test_bake, type = 'response')))
    print("pred ok")
    GAM_preds$label <- test_bake$label
    #AUC
    g <- roc(label ~ X1, data = GAM_preds, direction = "<")
    AUC <- g$auc
    metric_results[nrow(metric_results) + 1,] = list(dataset_vector[dataset_counter], i, "GAM", AUC)
    print(AUC)
    
    #####
    # LDA #needs CFS step_corr tidy
    #####

    nr_features_to_eliminate <- (sum(abs(cor(train_bake_x))>0.75)-ncol(train_bake_x))/2
    features_ranked <- attrEval(label~., train_bake, "MDL")
    selected_features <- names(features_ranked[-((ncol(train_bake_x)-nr_features_to_eliminate):ncol(train_bake_x))])
    train_bake_selected <- train_bake %>% dplyr::select(all_of(selected_features), label)
    train_bake_x_selected <- train_bake_x %>% dplyr::select(all_of(selected_features))
    
    FS <- cfs(label ~., data = datasets[[4]])
    
    LDA_model <- lda(label~., train_bake)
    LDA_preds <- data.frame(predict(LDA_model, test_bake, type = 'prob')$posterior)
    LDA_preds$label <- test_bake$label
    #AUC
    g <- roc(label ~ X1, data = LDA_preds, direction = "<")
    AUC <- g$auc
    metric_results[nrow(metric_results) + 1,] = list(dataset_vector[dataset_counter], i, "LDA", AUC)
    print(AUC)
    
    
    #####
    # QDA #needs CFS
    #####
    LDA_model <- train(data.frame(dplyr::select(train_bake,-label)), train_bake$label, method="stepLDA", trControl=trainControl(method="cv", number = 2, classProbs = TRUE, summaryFunction = BigSummary), metric = "partialGini",
                       #tuneGrid = expand.grid(maxvar = (floor(3*ncol(train)/4):ncol(train)), direction = "forward"),
                       allowParallel = TRUE)
    #works but suboptimal
    LDA_model <- lda(label~., data = train_bake)
    LDA_preds <- data.frame(predict(LDA_model, test_bake, type = 'prob')$posterior)
    LDA_preds$label <- test_bake$label
    #AUC
    g <- roc(label ~ X1, data = LDA_preds, direction = "<")
    AUC <- g$auc
    metric_results[nrow(metric_results) + 1,] = list(dataset_vector[dataset_counter], i, "LDA", AUC)
    print(AUC)
    
    
    
    
    
    #####
    # CTREE
    #####
    
    set.seed(innerseed)
    CTREE_model <- train(TREE_recipe, data = train, method = "ctree", trControl = ctrl,
          tuneGrid = expand.grid(mincriterion = hyperparameters_CTREE$mincriterion),
          metric = "Brier", maximize = FALSE)
    
    CTREE_preds <- predict(CTREE_model, test, type = 'probs')
    CTREE_preds$label <- test$label
    #AUC
    g <- roc(label ~ X1, data = CTREE_preds, direction = "<")
    AUC <- g$auc
    metric_results[nrow(metric_results) + 1,] = list(dataset_vector[dataset_counter], i, "CTREE", AUC)
    print(AUC)
    
    #PG
    pg <- partialGini(CTREE_preds$X1, CTREE_preds$label, 0.4)
    pg
    
    #Brier
    bs <- mean(((as.numeric(CTREE_preds$label)-1) - CTREE_preds$X1)^2)
    bs
  
    
    #####
    # RF
    #####
    modellist <- list()
    for (ntree in hyperparameters_RF$ntrees){
      set.seed(innerseed)
      RF_model <- train(TREE_recipe, data = train, method = "rf", trControl = ctrl,
                        tuneGrid = expand.grid(mtry = hyperparameters_RF$mtry),
                        ntree = ntree, maximize = TRUE, allowParallel = TRUE)
      key <- toString(ntree)
      modellist[[key]] <- RF_model
    }
    
    
    #AUC
    RF_model_AUC <- extractBestModel(modellist, "AUCROC")
    RF_preds_AUC <- data.frame(predict(RF_model_AUC, test, type = 'probs'))
    names(RF_preds_AUC) <- c("X0", "X1")
    RF_preds_AUC$label <- test$label
    g <- roc(label ~ X1, data = RF_preds_AUC, direction = "<")
    AUC <- g$auc
    metric_results[nrow(metric_results) + 1,] = list(dataset_vector[dataset_counter], i, "RF", AUC)
    print(AUC)
    
    #PG
    RF_model_PG <- extractBestModel(modellist, "partialGini")
    RF_preds_PG <- data.frame(predict(RF_model_PG, test, type = 'probs'))
    names(RF_preds_PG) <- c("X0", "X1")
    RF_preds_PG$label <- test$label
    pg <- partialGini(RF_preds_PG$X1, RF_preds_PG$label, 0.4)
    pg
    
    #Brier
    RF_model_Brier <- extractBestModel(modellist, "Brier")
    RF_preds_Brier <- data.frame(predict(RF_model_Brier, test, type = 'probs'))
    names(RF_preds_Brier) <- c("X0", "X1")
    RF_preds_Brier$label <- test$label
    bs <- mean(((as.numeric(RF_preds_Brier$label)-1) - RF_preds_Brier$X1)^2)
    bs
    
    
    #####
    # XGB
    #####

    set.seed(innerseed)
    
    XGB_model <- train(label~., data = train_bake, method = "xgbTree", trControl = ctrl,
                      tuneGrid = expand.grid(nrounds = hyperparameters_XGB$nrounds,
                                             eta = hyperparameters_XGB$eta,
                                             gamma = hyperparameters_XGB$gamma,
                                             max_depth = hyperparameters_XGB$max_depth,
                                             colsample_bytree = hyperparameters_XGB$colsample_bytree,
                                             min_child_weight = hyperparameters_XGB$min_child_weight,
                                             subsample = hyperparameters_XGB$subsample), allowParallel = TRUE)

    XGB_preds <- predict(XGB_model, test_bake, type = 'prob')
    XGB_preds$label <- test$label
    #AUC
    g <- roc(label ~ X1, data = XGB_preds, direction = "<")
    AUC <- g$auc
    metric_results[nrow(metric_results) + 1,] = list(dataset_vector[dataset_counter], i, "XGB", AUC)
    print(AUC)
    
    #PG
    pg <- partialGini(XGB_preds$X1, XGB_preds$label, 0.4)
    pg
    
    #Brier
    bs <- mean(((as.numeric(XGB_preds$label)-1) - XGB_preds$X1)^2)
    bs
    
    
    
    
    #tidymodels
    # WERKT NOG NIET MET PG
    
    xgb_folds <- train %>% vfold_cv(v=5)
    
    xgb_model <- 
      parsnip::boost_tree(
        mode = "classification",
        trees = tune(),
        tree_depth = tune(),
        learn_rate = tune(),
        loss_reduction = tune()
      ) %>%
      set_engine("xgboost")
    
    xgb_wf <- workflow() %>%
      add_recipe(TREE_recipe) %>%
      add_model(xgb_model)
    
    
    
    
    #WRONG PROBABLY
    pg_2 <- new_prob_metric(
      direction = "maximize",
      fn = function(data, lev, model = NULL) {
        # Extract true labels and predicted probabilities
        truth <- as.numeric(data$truth == lev[1]) # Assuming lev[1] is the positive class
        prob <- data$.pred_1 # Assuming .pred_1 column contains probabilities for positive class
        
        sorted_indices <- order(estimate, decreasing = TRUE)
        sorted_probs <- estimate[sorted_indices]
        sorted_actuals <- prob[sorted_indices]
        
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
        } else {
            # Set partial Gini to NA if there are not enough cases for ROC calculation
            NA
        }
      }
    )
    
    
    metrics <- metric_set(roc_auc, brier_class)
    
    xgboost_tuned <- tune::tune_grid(
      object = xgb_wf,
      resamples = xgb_folds,
      grid = hyperparameters_XGB_tidy,
      metrics = metrics,
      control = tune::control_grid(verbose = TRUE)
    )
    
    xgboost_tuned %>%
      tune::show_best(metric = "roc_auc") %>%
      knitr::kable()
    
    best_booster <- xgboost_tuned %>% select_best("roc_auc")
    finalxgb__wf <- xgb_wf %>% finalize_workflow(best_booster)
    final_xgb_fit <- finalxgb__wf %>% last_fit(folds$splits[[i]])
    
    auc <- final_xgb_fit %>%
      collect_metrics()
    
    roc_auc_value <- final_xgb_fit %>%
      collect_metrics() %>%
      filter(.metric == "roc_auc") %>%
      pull(.estimate)
    

    final_xgb_fit %>%
      collect_predictions() %>% 
      roc_curve(label, .pred_X0) %>% 
      autoplot()

    #####
    # LightGBM
    #####
    
    lgbm_model <- 
      parsnip::boost_tree(
        mode = "classification",
        trees = tune(),
        tree_depth = tune(),
        learn_rate = tune(),
        loss_reduction = tune()
      ) %>%
      set_engine("lightgbm") %>%
      translate()
    
    lgbm_wf <- workflow() %>%
      add_recipe(TREE_recipe) %>%
      add_model(lgbm_model)
  
    lgbm_tuned <- tune::tune_grid(
      object = lgbm_wf,
      resamples = xgb_folds, #same folds as xgboost
      grid = hyperparameters_XGB_tidy, #same setting as xgboost
      metrics = metrics,
      control = tune::control_grid(verbose = TRUE)
    )
    
    lgbm_tuned %>%
      tune::show_best(metric = "brier_class") %>%
      knitr::kable()
    
    best_booster_auc <- lgbm_tuned %>% select_best("roc_auc")
    final_lgbm_wf_auc <- lgbm_wf %>% finalize_workflow(best_booster_auc)
    final_lgbm_fit_auc <- final_lgbm_wf_auc %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    best_booster_brier <- lgbm_tuned %>% select_best("brier_class")
    final_lgbm_wf_brier <- lgbm_wf %>% finalize_workflow(best_booster_brier)
    final_lgbm_fit_brier <- final_lgbm_wf_brier %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    
    auc <- final_lgbm_fit_auc %>%
      collect_metrics() %>%
      filter(.metric == "roc_auc") %>%
      pull(.estimate)
    brier <- final_lgbm_fit_brier %>%
      collect_metrics() %>%
      filter(.metric == "brier_class") %>%
      pull(.estimate)
    
    
    roc_auc_value <- final_lgbm_fit %>%
      collect_metrics() %>%
      filter(.metric == "roc_auc") %>%
      pull(.estimate)
    
    
    final_lgbm_fit %>%
      collect_predictions() %>% 
      roc_curve(label, .pred_X0) %>% 
      autoplot()
    
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