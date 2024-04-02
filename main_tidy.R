# main
if (!require("pacman")) install.packages("pacman") ; require("pacman")
p_load(glmnet, glmnetUtils, mgcv, tidyverse, xgboost, DiagrammeR, stringr, tictoc, doParallel, pROC, earth, Matrix, pre, caret, parsnip, ggplot2, recipes, rsample, workflows, healthyR.ai, rlang, yardstick, bonsai, lightgbm, ranger, tune, DescTools)
#conventions:
# target variable = label
# indepentent variables = all others

#levels should be X1 and X2
#evaluation function for caret tuning from pre package
source("./src/models.R")
#source("./src/partialGini_yardstick.R")
source("./src/hyperparameters.R")
source("./src/BigSummary.R")
source("./src/data_loader.R")
datasets <- load_data()
cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)

#(AUCROC, Brier, partialGini)
metric = "AUCROC"
nr_repeats = 3
outerfolds = 2
innerfolds = 3
dataset_vector = c("GC", "AC", "GMSC", "TH02")

ctrl <- trainControl(method = "cv", number = innerfolds, classProbs = TRUE, summaryFunction = BigSummary, search = "grid", allowParallel = TRUE)
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
  innerfolds = nr_repeats
  
  # dummies for categories present in train and test so no new levels error
  MINIMAL_recipe <- recipe(label ~., data = dataset) %>%
    step_zv(all_predictors()) %>%
    step_dummy(all_string_predictors()) %>%
    step_dummy(all_factor_predictors())
  
  dataset <- MINIMAL_recipe %>% prep() %>% bake(dataset)                        #dit misschien trainen?

  set.seed(111)
  # create 5x2 folds
  folds <- vfold_cv(dataset, v = outerfolds, repeats = nr_repeats, strata = NULL)
  for(i in 1:nrow(folds)) { #CHANGE
    cat("Fold", i, "/ 10 \n")
    train <- analysis(folds$splits[[i]])
    test <- assessment(folds$splits[[i]])
    
    # formulate recipes
    # dummies for categories present in train and test so no new levels error
    #MINIMAL_recipe <- recipe(label ~., data = rbind(train, test)) %>%
    #  step_zv(all_predictors()) %>%
    #  step_dummy(all_string_predictors()) %>%
    #  step_dummy(all_factor_predictors())
    #
    #train <- MINIMAL_recipe %>% prep() %>% bake(folds)
    #test <- MINIMAL_recipe %>% prep() %>% bake(test)
    
    # for tree-based that don't require dummies
    TREE_recipe <- recipe(label ~., data = train) %>%
      step_impute_mean(all_numeric_predictors()) %>%
      step_impute_mode(all_string_predictors()) %>%
      step_impute_mode(all_factor_predictors()) %>%
      step_zv()
    
    # for tree-based that do require dummies
    XGB_recipe <- recipe(label ~., data = train) %>%
      step_impute_mean(all_numeric_predictors()) %>%
      step_impute_mode(all_string_predictors()) %>%
      step_impute_mode(all_factor_predictors()) %>%
      step_dummy(all_string_predictors()) %>%
      step_dummy(all_factor_predictors()) %>%
      step_zv()
    
    winsorizable <- names(train)[get_splineworthy_columns(train)]
    
    # for algorithms using linear terms (LRR, gam, rule ensembles)
    LINEAR_recipe <- recipe(label ~., data = train) %>%
      step_impute_mean(all_numeric_predictors()) %>%
      step_impute_mode(all_string_predictors()) %>%
      step_impute_mode(all_factor_predictors()) %>%
      step_hai_winsorized_truncate(all_numeric_predictors(), fraction = 0.025) %>%
      step_rm(!contains("winsorized") & all_numeric_predictors()) %>%
      step_dummy(all_string_predictors()) %>%
      step_dummy(all_factor_predictors()) %>%
      step_zv(all_predictors()) %>%
      step_normalize(all_numeric_predictors())
    
    
    RULEFIT_recipe <- TREE_recipe
    

    innerseed <- i
    
    ############################################################################
    # SINGLE CLASSIFIERS
    ############################################################################
    
    #####
    # LRR
    #####
    print("LRR")
    
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
      resamples = inner_folds,
      grid = hyperparameters_LR_R_tidy, 
      metrics = metrics,
      control = tune::control_grid(verbose = TRUE, save_pred = TRUE)
    )
    
    best_model_auc <- LRR_tuned %>% select_best("roc_auc")
    final_LRR_wf_auc <- LRR_wf %>% finalize_workflow(best_model_auc)
    final_LRR_fit_auc <- final_LRR_wf_auc %>% last_fit(folds$splits[[i]], metrics = metrics)

    best_model_brier <- LRR_tuned %>% select_best("brier_class")
    final_LRR_wf_brier <- LRR_wf %>% finalize_workflow(best_model_brier)
    final_LRR_fit_brier <- final_LRR_wf_brier %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    best_model_pg <- LRR_tuned %>% select_best_pg_LRR()
    final_LRR_wf_pg <- LRR_wf %>% finalize_workflow(best_model_pg)
    final_LRR_fit_pg <- final_LRR_wf_pg %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    
    
    auc <- final_LRR_fit_auc %>%
      collect_metrics() %>%
      filter(.metric == "roc_auc") %>%
      pull(.estimate)
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "LRR", auc)
    
    brier <- final_LRR_fit_brier %>%
      collect_metrics() %>%
      filter(.metric == "brier_class") %>%
      pull(.estimate)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "LRR", brier)

    pg <- final_LRR_fit_pg %>%
      collect_pg()
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "LRR", pg)
    
    #####
    # GAM
    #####
    print("GAM")
    
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
    GAM_preds$label <- test_bake$label
    #AUC
    g <- roc(label ~ X1, data = GAM_preds, direction = "<")
    AUC <- g$auc
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "GAM", AUC)
    #Brier
    brier <- brier_class_vec(GAM_preds$label, GAM_preds$X1)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "GAM", brier)
    #PG
    pg <- partialGini(GAM_preds$X1, GAM_preds$label)
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "GAM", pg)

    
    #####
    # LDA #needs CFS step_corr tidy
#    #####
#    print("LDA")
#  #step_corr doesnt work?
#    nr_features_to_eliminate <- (sum(abs(cor(train_bake_x))>0.75)-ncol(train_bake_x))/2
#    features_ranked <- attrEval(label~., train_bake, "MDL")
#    selected_features <- names(features_ranked[-((ncol(train_bake_x)-nr_features_to_eliminate):ncol(train_bake_x))])
#    corr_recipe <- recipe(label~., train_bake) %>%
#      step_corr(threshold = 0.1) %>%
#      prep()
#    train_bake_selected <- corr_recipe %>% bake(train_bake)
#    test_bake_selected <- corr_recipe %>% bake(test_bake)
#    %>% dplyr::select(all_of(selected_features), label)
#    train_bake_x_selected <- train_bake_x %>% dplyr::select(all_of(selected_features))
#    
#    FS <- cfs(label ~., data = datasets[[4]])
#    
#    LDA_model <- lda(label~., train_bake)
#    LDA_preds <- data.frame(predict(LDA_model, test_bake, type = 'prob')$posterior)
#    LDA_preds$label <- test_bake$label
#    #AUC
#    g <- roc(label ~ X1, data = LDA_preds, direction = "<")
#    AUC <- g$auc
#    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "LDA", AUC)
#    print(AUC)
#    
#    brier <- brier_class_vec(LDA_preds$label, LDA_preds$X1)
#    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "LDA", brier)
#    
#    
#    #####
#    # QDA #needs CFS
#    #####
#    print("QDA")
    
#    QDA_model <- train(data.frame(dplyr::select(train_bake,-label)), train_bake$label, method="stepLDA", trControl=trainControl(method="cv", number = 2, classProbs = TRUE, summaryFunction = BigSummary), metric = "partialGini",
#                       #tuneGrid = expand.grid(maxvar = (floor(3*ncol(train)/4):ncol(train)), direction = "forward"),
#                       allowParallel = TRUE)
#    #works but suboptimal
#    LDA_model <- lda(label~., data = train_bake)
#    LDA_preds <- data.frame(predict(LDA_model, test_bake, type = 'prob')$posterior)
#    LDA_preds$label <- test_bake$label
#    #AUC
#    g <- roc(label ~ X1, data = LDA_preds, direction = "<")
#    AUC <- g$auc
#    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "LDA", AUC)
#    print(AUC)
    
    
    
    
    #####
    # CTREE
    #####
    print("CTREE")    
    set.seed(innerseed)
    CTREE_model <- train(TREE_recipe, data = train, method = "ctree", trControl = ctrl,
          tuneGrid = expand.grid(mincriterion = hyperparameters_CTREE$mincriterion),
          metric = "AUCROC")

    #AUC
    CTREE_preds <- predict(CTREE_model, test, type = 'prob')
    CTREE_preds$label <- test$label
    
    g <- roc(label ~ X1, data = CTREE_preds, direction = "<")
    AUC <- g$auc
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "CTREE", AUC)
    #Brier
    
    CTREE_model_Brier <- train(TREE_recipe, data = train, method = "ctree",
                               tuneGrid = expand.grid(mincriterion = (CTREE_model$results%>%slice_max(Brier)%>%select(mincriterion))[[1]])) 
    CTREE_preds <- predict(CTREE_model_Brier, test, type = 'prob')
    CTREE_preds$label <- test$label
    
    
    brier <- brier_class_vec(CTREE_preds$label, CTREE_preds$X1)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "CTREE", brier)
    
    #PG
    
    CTREE_model_PG <- train(TREE_recipe, data = train, method = "ctree",
                               tuneGrid = expand.grid(mincriterion = (CTREE_model$results%>%slice_max(partialGini)%>%select(mincriterion))[[1]]))
    CTREE_preds <- predict(CTREE_model_PG, test, type = 'prob')
    CTREE_preds$label <- test$label
    
    pg <- partialGini(CTREE_preds$X1, CTREE_preds$label)
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "CTREE", pg)
    
    ############################################################################
    # HOMOGENEOUS ENSEMBLES
    ############################################################################
    
    #####
    # RF
    #####
    print("RF")
    
    #reload hyperparameters because it uses ncol(train_bake_x)
    source("./src/hyperparameters.R")
    
    train_RF <- train
    test_RF <- test
    
#    if(dataset_counter==1) {
#      # Find the three levels with the lowest frequency
#
#      lowest_levels_X4 <- names(sort(table(train$X4))[1:4])
#      
#      # Combine the three lowest levels into a new level, for example, "Other"
#      train_RF$X4 <- factor(ifelse(train_RF$X4 %in% lowest_levels_X4, "Other", as.character(train_RF$X4)))
#      test_RF$X4 <- factor(ifelse(test_RF$X4 %in% lowest_levels_X4, "Other", as.character(test_RF$X4)))
#    }
#      RF_recipe <- train_RF %>% recipe(label~.) %>% 
#        step_impute_mean(all_numeric_predictors()) %>%
#        step_impute_mode(all_string_predictors()) %>%
#        step_impute_mode(all_factor_predictors()) %>%
#        step_zv(all_predictors())
#      train_RF_bake <- RF_recipe %>%
#        prep() %>%
#        bake(train_RF)
#      test_RF_bake <- RF_recipe %>%
#        prep() %>%
#        bake(test_RF)
#    
    
    modellist <- list()
    for (ntree in c(100,250,500,750,1000)){
      set.seed(innerseed)
      print(ntree)
      
      RF_model <- train(TREE_recipe, data = train, method = "ranger", trControl = ctrl,
                        tuneGrid = expand.grid(mtry = hyperparameters_RF$mtry,
                                               splitrule = hyperparameters_RF$splitrule,
                                               min.node.size = hyperparameters_RF$min.node.size),
                        num.tree = ntree, 
                        metric = "AUCROC")
      key <- toString(ntree)
      modellist[[key]] <- RF_model
    }
    
    
    #AUC
    RF_model_AUC <- extractBestModel(modellist, "AUCROC")
    RF_preds_AUC <- data.frame(predict(RF_model_AUC, test_RF, type = 'prob'))
    names(RF_preds_AUC) <- c("X0", "X1")
    RF_preds_AUC$label <- test_RF$label
    g <- roc(label ~ X1, data = RF_preds_AUC, direction = "<")
    AUC <- g$auc
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "RF", AUC)
    
    #PG
    RF_model_PG <- extractBestModel(modellist, "partialGini")
    RF_preds_PG <- data.frame(predict(RF_model_PG, test_RF, type = 'prob'))
    names(RF_preds_PG) <- c("X0", "X1")
    RF_preds_PG$label <- test_RF$label
    pg <- partialGini(RF_preds_PG$X1, RF_preds_PG$label)
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "RF", pg)
    
    #Brier
    RF_model_Brier <- extractBestModel(modellist, "Brier")
    RF_preds_Brier <- data.frame(predict(RF_model_Brier, test_RF, type = 'prob'))
    names(RF_preds_Brier) <- c("X0", "X1")
    RF_preds_Brier$label <- test_RF$label
    bs <- mean(((as.numeric(RF_preds_Brier$label)-1) - RF_preds_Brier$X1)^2)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "RF", bs)
    
    
    #####
    # XGB
    #####
    print("XGB")

    set.seed(innerseed)
    
#    XGB_model <- train(TREE_recipe, data = train, method = "xgbTree", trControl = ctrl,
#                      tuneGrid = expand.grid(nrounds = hyperparameters_XGB$nrounds,
#                                             eta = hyperparameters_XGB$eta,
#                                             gamma = hyperparameters_XGB$gamma,
#                                             max_depth = hyperparameters_XGB$max_depth,
#                                             colsample_bytree = hyperparameters_XGB$colsample_bytree,
#                                             min_child_weight = hyperparameters_XGB$min_child_weight,
#                                             subsample = hyperparameters_XGB$subsample),
#                      allowParallel = TRUE
#                      )
#
#    XGB_preds <- predict(XGB_model, test, type = 'prob')
#    XGB_preds$label <- test$label
#    #AUC
#    best_model_auc <- getModelInfo("glmnet")$grid %>%
#      filter(AUC == max(AUC)) %>%
#      slice(1) %>%
#      .$model
#    
#    
#    
#    
#    g <- roc(label ~ X1, data = XGB_preds, direction = "<")
#    AUC <- g$auc
#    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "XGB", AUC)
#    print(AUC)
#   
#    #PG
#    pg <- partialGini(XGB_preds$X1, XGB_preds$label, 0.4)
#    pg
#   
#    #Brier
#    bs <- mean(((as.numeric(XGB_preds$label)-1) - XGB_preds$X1)^2)
#    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "XGB", bs)
#    bs
#   
  
   
   
    #tidymodels
    # WERKT NOG NIET MET PG
    
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
      add_recipe(XGB_recipe) %>%
      add_model(xgb_model)
    print(hyperparameters_XGB_tidy)
    
    
    xgb_tuned <- tune::tune_grid(
      object = xgb_wf,
      resamples = inner_folds,
      grid = hyperparameters_XGB_tidy,
      metrics = metrics,
      control = tune::control_grid(verbose = TRUE, save_pred = TRUE)
    )

    
    best_booster_auc <- xgb_tuned %>% select_best("roc_auc")
    final_xgb_wf_auc <- xgb_wf %>% finalize_workflow(best_booster_auc)
    final_xgb_fit_auc <- final_xgb_wf_auc %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    best_booster_brier <- xgb_tuned %>% select_best("brier_class")
    final_xgb_wf_brier <- xgb_wf %>% finalize_workflow(best_booster_brier)
    final_xgb_fit_brier <- final_xgb_wf_brier %>% last_fit(folds$splits[[i]], metrics = metrics)

    best_booster_pg <- xgb_tuned %>% select_best_pg_XGB()
    final_xgb_wf_pg <- xgb_wf %>% finalize_workflow(best_booster_pg)
    final_xgb_fit_pg <- final_xgb_wf_pg %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    
    auc <- final_xgb_fit_auc %>%
      collect_metrics() %>%
      filter(.metric == "roc_auc") %>%
      pull(.estimate)
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "XGB", auc)
    
    brier <- final_xgb_fit_brier %>%
      collect_metrics() %>%
      filter(.metric == "brier_class") %>%
      pull(.estimate)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "XGB", brier)
    
    pg <- final_xgb_fit_pg %>%
      collect_pg()
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "XGB", pg)

    #####
    # LightGBM
    #####
    print("LGBM")
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
      resamples = inner_folds, 
      grid = hyperparameters_XGB_tidy, #same setting as xgboost
      metrics = metrics,
      control = tune::control_grid(verbose = TRUE, save_pred = TRUE)
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
    
    best_model_pg <- lgbm_tuned %>% select_best_pg_XGB()
    final_lgbm_wf_pg <- lgbm_wf %>% finalize_workflow(best_model_pg)
    final_lgbm_fit_pg <- final_lgbm_wf_pg %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    auc <- final_lgbm_fit_auc %>%
      collect_metrics() %>%
      filter(.metric == "roc_auc") %>%
      pull(.estimate)
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "LGBM", auc)
    
    brier <- final_lgbm_fit_brier %>%
      collect_metrics() %>%
      filter(.metric == "brier_class") %>%
      pull(.estimate)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "LGBM", brier)
    
    pg <- final_xgb_fit_pg %>%
      collect_pg()
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "LGBM", pg)
    
    ############################################################################
    # HETEROGENEOUS (RULE) ENSEMBLES 
    ############################################################################
    
    #####
    # RE
    #####
    print("RE")
    train_RE <- train
    test_RE <- test
    
#    if(dataset_counter==1) {
#      # Find the three levels with the lowest frequency
#
#      lowest_levels_X4 <- names(sort(table(train$X4))[1:4])
#      lowest_levels_X17 <- names(sort(table(train$X17))[1:2])
#      
#      # Combine the three lowest levels into a new level, for example, "Other"
#      train_RE$X4 <- factor(ifelse(train_RE$X4 %in% lowest_levels_X4, "Other", as.character(train_RE$X4)))
#      test_RE$X4 <- factor(ifelse(test_RE$X4 %in% lowest_levels_X4, "Other", as.character(test_RE$X4)))
#      train_RE$X17 <- factor(ifelse(train_RE$X17 %in% lowest_levels_X17, "Other", as.character(train_RE$X17)))
#      test_RE$X17 <- factor(ifelse(test_RE$X17 %in% lowest_levels_X17, "Other", as.character(test_RE$X17)))
#    }
      RE_recipe <- train_RE %>% recipe(label~.) %>% 
        step_impute_mean(all_numeric_predictors()) %>%
        step_impute_mode(all_string_predictors()) %>%
        step_impute_mode(all_factor_predictors()) %>%
        step_zv(all_predictors())
      train_RE_baked <- RE_recipe %>%
        prep() %>%
        bake(train_RE)
      test_RE_baked <- RE_recipe %>%
        prep() %>%
        bake(test_RE)
    
    
    #ntrees = 50: start 18:08, final rules created 18:18, done 18:20
    #ntrees = 5: start 18:21, final rules 18:22, done 18:24
    set.seed(innerseed)
    RE_model <- train(XGB_recipe, data = train, method = "pre",
                      ntrees = 500, family = "binomial", trControl = ctrl,
                      tuneGrid = preGrid, ad.alpha = 0, singleconditions = TRUE,
                      winsfrac = 0.05, normalize = TRUE, #same a priori influence as a typical rule
                      verbose = TRUE,
                      metric = "AUCROC", allowParallel = TRUE,
                      par.init=TRUE,
                      par.final=TRUE)
    RE_learning_rate <- RE_model$bestTune$learnrate
    
    
    
    
    
    
    
    
    #AUC
    RE_preds <- predict(RE_model, test_RE, type = 'probs')
    RE_preds$label <- test_RE$label
    
    g <- roc(label ~ X1, data = RE_preds, direction = "<")
    AUC <- g$auc
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "RE", AUC)
    
    #Brier
    RE_model_Brier <- train(XGB_recipe, data = train, method = "pre",
                            ntrees = 500, family = "binomial", trControl = trainControl(method = "none", classProbs = TRUE),
                            tuneGrid = getModelInfo("pre")[[1]]$grid( 
                              maxdepth = (RE_model$results%>%slice_max(Brier)%>%select(maxdepth))[[1]],
                              learnrate = (RE_model$results%>%slice_max(Brier)%>%select(learnrate))[[1]],
                              penalty.par.val = c("lambda.1se"), # λand γ combination yielding the sparsest solution within 1 standard error of the error criterion of the minimum is returned
                              sampfrac = 1,
                              use.grad = TRUE), ad.alpha = 0, singleconditions = TRUE,
                            winsfrac = 0.05, normalize = TRUE, #same a priori influence as a typical rule
                            verbose = TRUE,
                            allowParallel = TRUE,
                            par.init=TRUE,
                            par.final=TRUE)
    RE_preds <- predict(RE_model_Brier, test, type = 'prob')
    RE_preds$label <- test$label
    brier <- brier_class_vec(RE_preds$label, RE_preds$X1)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "RE", brier)
    
    #PG
    RE_model_PG <- train(XGB_recipe, data = train, method = "pre",
                            ntrees = 500, family = "binomial", trControl = trainControl(method = "none", classProbs = TRUE),
                            tuneGrid = getModelInfo("pre")[[1]]$grid( 
                              maxdepth = (RE_model$results%>%slice_max(partialGini)%>%select(maxdepth))[[1]],
                              learnrate = (RE_model$results%>%slice_max(partialGini)%>%select(learnrate))[[1]],
                              penalty.par.val = c("lambda.1se"), # λand γ combination yielding the sparsest solution within 1 standard error of the error criterion of the minimum is returned
                              sampfrac = 1,
                              use.grad = TRUE), ad.alpha = 0, singleconditions = TRUE,
                            winsfrac = 0.05, normalize = TRUE, #same a priori influence as a typical rule
                            verbose = TRUE,
                            metric = "AUCROC", allowParallel = TRUE,
                            par.init=TRUE,
                            par.final=TRUE)
    RE_preds <- predict(RE_model_PG, test, type = 'prob')
    RE_preds$label <- test$label
    
    pg <- partialGini(RE_preds$X1, RE_preds$label)
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "RE", pg)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    print("RE")
    RE_model <- 
      parsnip::rule_fit(
        mode = "classification",
        trees = 500,
        tree_depth = tune(),
        learn_rate = tune(),
        penalty = tune()
        ) %>%
      set_engine("xrf") %>%
      translate()
    
    RE_wf <- workflow() %>%
      add_recipe(XGB_recipe) %>%
      add_model(RE_model)
    
    RE_tuned <- tune::tune_grid(
      object = RE_wf,
      resamples = inner_folds, 
      grid = expand.grid( 
        tree_depth = c(2,3,4),
        learn_rate = c(.01, .05, .1),
        penalty = c(.01, .05, .1, .2)
        #mtry = sqrt(ncol(train_bake_x)*c(0.1,0.25,0.5,1,2,4)),
      ), #same setting as xgboost
      metrics = metrics,
      control = tune::control_grid(verbose = TRUE, save_pred = TRUE)
    )
    
    RE_tuned %>%
      tune::show_best(metric = "brier_class") %>%
      knitr::kable()
    
    best_booster_auc <- RE_tuned %>% select_best("roc_auc")
    final_RE_wf_auc <- RE_wf %>% finalize_workflow(best_booster_auc)
    final_RE_fit_auc <- final_RE_wf_auc %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    best_booster_brier <- RE_tuned %>% select_best("brier_class")
    final_RE_wf_brier <- RE_wf %>% finalize_workflow(best_booster_brier)
    final_RE_fit_brier <- final_RE_wf_brier %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    best_model_pg <- RE_tuned %>% select_best_pg_RE()
    final_RE_wf_pg <- RE_wf %>% finalize_workflow(best_model_pg)
    final_RE_fit_pg <- final_RE_wf_pg %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    auc <- final_RE_fit_auc %>%
      collect_metrics() %>%
      filter(.metric == "roc_auc") %>%
      pull(.estimate)
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "RE", auc)
    
    brier <- final_RE_fit_brier %>%
      collect_metrics() %>%
      filter(.metric == "brier_class") %>%
      pull(.estimate)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "RE", brier)
    
    pg <- final_RE_fit_pg %>%
      collect_pg()
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "RE", pg)
    
    
    
    
    
    
    
    
    
    #####
    # HRE
    #####
    #print("HRE")
    #set.seed(innerseed)
    #
    #if(dataset_counter==1) {
    #  # Find the three levels with the lowest frequency
    #  train_HRE <- train
    #  test_HRE <- test
    #  lowest_levels_X4 <- names(sort(table(train$X4))[1:4])
    #  lowest_levels_X17 <- names(sort(table(train$X17))[1:2])
    #  
    #  # Combine the three lowest levels into a new level, for example, "Other"
    #  train_HRE$X4 <- factor(ifelse(train_HRE$X4 %in% lowest_levels_X4, "Other", as.character(train_HRE$X4)))
    #  test_HRE$X4 <- factor(ifelse(test_HRE$X4 %in% lowest_levels_X4, "Other", as.character(test_HRE$X4)))
    #  train_HRE$X17 <- factor(ifelse(train_HRE$X17 %in% lowest_levels_X17, "Other", as.character(train_HRE$X17)))
    #  test_HRE$X17 <- factor(ifelse(test_HRE$X17 %in% lowest_levels_X17, "Other", as.character(test_HRE$X17)))
    #  HRE_recipe <- train_HRE %>% recipe(label~.) %>% 
    #    step_impute_mean(all_numeric_predictors()) %>%
    #    step_impute_mode(all_string_predictors()) %>%
    #    step_impute_mode(all_factor_predictors()) %>%
    #    step_zv(all_predictors()) %>%
    #    step_dummy(all_string_predictors()) %>%
    #    step_dummy(all_factor_predictors()) %>%
    #    step_zv(all_predictors())
    #  train_HRE_baked <- HRE_recipe %>%
    #    prep() %>%
    #    bake(train_HRE)
    #  test_HRE_baked <- HRE_recipe %>%
    #    prep() %>%
    #    bake(test_HRE)
    #}
    #  
    #set.seed(innerseed)
    #HRE_model <- gpe(label ~., data = train_HRE_baked,
    #                 base_learners = list(gpe_trees(learnrate = RE_model$bestTune$learnrate, ntrees = 500),#learn rate based on AUC
    #                                      gpe_earth(degree = 3, nk = 2*sum(get_splineworthy_columns(train_HRE_baked))),
    #                                      gpe_linear()),
    #                 penalized_trainer = gpe_cv.glmnet(family = "binomial", ad.alpha = 0, weights = NULL))
    #
    #HRE_preds <- data.frame(predict(HRE_model, test_HRE_baked, type = 'response'))
    #HRE_preds$label <- test_HRE_baked$label
    ##AUC
    #g <- roc(label ~ lambda.1se, data = HRE_preds, direction = "<")
    #AUC <- g$auc
    #AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "HRE", AUC)
    ##Brier
    #brier <- brier_class_vec(HRE_preds$label, HRE_preds$lambda.1se)
    #Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "HRE", brier)
    #
    ##PG
    #pg <- partialGini(HRE_preds$X1, HRE_preds$label)
    #PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "HRE", pg)
    #
    #####
    # SRE
    #####
    print("SRE")
    #extract smooth term names for SRE
    smooth_terms <- grep("s\\(", unlist(str_split(as.character(formula(GAM_model))[3], " \\+ ")), value = TRUE)
    # Extract and fitted values for each smooth term
    fitted_smooths_train <- data.frame(matrix(ncol = length(smooth_terms), nrow = nrow(train_bake)))
    fitted_smooths_test <- data.frame(matrix(ncol = length(smooth_terms), nrow = nrow(test_bake)))
    colnames(fitted_smooths_train) <- smooth_terms
    colnames(fitted_smooths_test) <- smooth_terms
    for (j in seq_along(smooth_terms)) {
      current_smooth <- smooth_terms[j]
      fitted_values_train <- predict(GAM_model, type = "terms")[, current_smooth]
      fitted_smooths_train[, j] <- fitted_values_train
      fitted_values_test <- predict(GAM_model, test_bake, type = "terms")[, current_smooth]
      fitted_smooths_test[, j] <- fitted_values_test 
    }
    
    train_rule_baked <- XGB_recipe %>%
      prep(train) %>%
      bake(train)
    test_rule_baked <- XGB_recipe %>%
      prep(train) %>%
      bake(test)    
    SRE_train_rules <- fit_rules(train_rule_baked, drop_na(tibble(rules = coef(extract_fit_engine(final_RE_fit_auc))$rule))$rules)
    SRE_test_rules <- fit_rules(test_rule_baked, drop_na(tibble(rules = coef(extract_fit_engine(final_RE_fit_auc))$rule))$rules)
    
    SRE_train <- cbind(SRE_train_rules, fitted_smooths_train)
    SRE_test <- cbind(SRE_test_rules, fitted_smooths_test)
    

    
    #adalasso with initial ridge weights
    #INITIAL RIDGE
#
#    set.seed(innerseed)
#    inner_folds_SRE <- SRE_train %>% vfold_cv(v=5)
#    ridge1 <- 
#      parsnip::logistic_reg(
#        mode = "classification",
#        mixture = tune(),
#        penalty = tune()
#        ) %>%
#      set_engine("glmnet")
#    
#    LRR_wf <- workflow() %>%
#      add_formula(formula = label ~.) %>%
#      add_model(ridge1)
#    
#    LRR_tuned <- tune::tune_grid(
#      object = LRR_wf,
#      resamples = inner_folds_SRE,
#      grid = expand.grid(list(mixture = 0, penalty = seq(0.001,5,length=100))),
#      metrics = metrics,
#      control = tune::control_grid(verbose = TRUE, save_pred = TRUE)
#    )
#    
#    best_model_auc <- LRR_tuned %>% select_best("roc_auc")
#    final_LRR_wf_auc <- LRR_wf %>% finalize_workflow(best_model_auc) %>% fit(SRE_train)
#    auc_ridge_coef <- final_LRR_wf_auc%>%extract_fit_parsnip()%>%tidy()%>%select(estimate)
#
#    best_model_brier <- LRR_tuned %>% select_best("brier_class")
#    final_LRR_wf_brier <- LRR_wf %>% finalize_workflow(best_model_brier) %>% fit(SRE_train)
#    brier_ridge_coef <- final_LRR_wf_brier%>%extract_fit_parsnip()%>%tidy()%>%select(estimate)
#    
#    best_model_pg <- LRR_tuned %>% select_best_pg()
#    final_LRR_wf_pg <- LRR_wf %>% finalize_workflow(best_model_pg) %>% fit(SRE_train)
#    pg_ridge_coef <- final_LRR_wf_pg%>%extract_fit_parsnip()%>%tidy()%>%select(estimate)
#    
#    ## Perform adaptive LASSO
#
#    auc_penalties <- c(1/abs(auc_ridge_coef$estimate[-1]))
#    
#    adaLasso_auc <- 
#      parsnip::logistic_reg(
#        mode = "classification",
#        mixture = tune(),
#        penalty = tune()
#      ) %>%
#      set_engine("glmnet", penalty.factor = auc_penalties)
#    
#    adaLasso_wf_auc <- workflow() %>%
#      add_formula(formula = label ~.) %>%
#      add_model(adaLasso_auc)
#    
#    adaLasso_brier <- 
#      parsnip::logistic_reg(
#        mode = "classification",
#        mixture = tune(),
#        penalty = tune()
#      ) %>%
#      set_engine("glmnet", penalty.factor = 1/abs(brier_ridge_coef$estimate[-1]))
#    
#    adaLasso_wf_brier <- workflow() %>%
#      add_formula(formula = label ~.) %>%
#      add_model(adaLasso_brier)
#    
#    adaLasso_pg <- 
#      parsnip::logistic_reg(
#        mode = "classification",
#        mixture = tune(),
#        penalty = tune()
#      ) %>%
#      set_engine("glmnet"#, penalty.factor = 1/abs(pg_ridge_coef$estimate[-1]))
#      )
#    
#    adaLasso_wf_pg <- workflow() %>%
#      add_formula(formula = label ~.) %>%
#      add_model(adaLasso_pg)
#    
#    adaLasso_tuned_auc <- tune::tune_grid(
#      object = adaLasso_wf_auc,
#      resamples = inner_folds_SRE,
#      grid = expand.grid(list(mixture = 1, penalty = seq(0.001,0.1,length=100))), 
#      metrics = metrics,
#      control = tune::control_grid(verbose = TRUE)
#    )
#    
#    adaLasso_tuned_brier <- tune::tune_grid(
#      object = adaLasso_wf_brier,
#      resamples = inner_folds_SRE,
#      grid = expand.grid(list(mixture = 1, penalty = seq(0.001,5,length=100))), 
#      metrics = metrics,
#      control = tune::control_grid(verbose = TRUE, save_pred = TRUE)
#    )
#    
#    adaLasso_tuned_pg <- tune::tune_grid(
#      object = adaLasso_wf_pg,
#      resamples = inner_folds_SRE,
#      grid = expand.grid(list(mixture = 1, penalty = seq(0.001,0.1,length=100))), 
#      metrics = metrics,
#      control = tune::control_grid(verbose = TRUE, save_pred = TRUE)
#    )
#    
#    #AUC
#    best_SRE_auc <- adaLasso_tuned_auc %>% select_best("roc_auc")
#    final_SRE_wf_auc <- adaLasso_wf_auc %>% finalize_workflow(best_SRE_auc)
#    SRE_auc_preds <- final_SRE_wf_auc %>% fit(SRE_train) %>% predict(SRE_test, type = "prob")
#    
#    SRE_auc_preds$label <- SRE_test$label
#    #AUC
#    g <- roc(label ~ .pred_X1, data = SRE_auc_preds, direction = "<")
#    AUC <- g$auc
#    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE", AUC)
#
#    #Brier
#    best_SRE_brier <- adaLasso_tuned_brier %>% select_best("brier_class")
#    final_SRE_wf_brier <- adaLasso_wf_brier %>% finalize_workflow(best_SRE_brier)
#    SRE_brier_preds <- final_SRE_wf_brier%>% fit(SRE_train) %>% predict(SRE_test, type = "prob")
#    
#    SRE_brier_preds$label <- SRE_test$label
#    brier <- brier_class_vec(SRE_brier_preds$label, SRE_brier_preds$.pred_X1)
#    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE", brier)
#    
#    #PG
#    best_SRE_pg <- adaLasso_tuned_pg %>% select_best_pg()
#    final_SRE_wf_pg <- adaLasso_wf_pg %>% finalize_workflow(best_SRE_pg)
#    SRE_pg_preds <- final_SRE_wf_pg %>% fit(SRE_train) %>% predict(SRE_test, type = "prob")
#    
#    SRE_pg_preds$label <- SRE_test$label
#    pg <- partialGini(SRE_pg_preds$.pred_X1, SRE_pg_preds$label)
#    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE", pg)
 
    
    
    ###################
    
    
    
       
    #without parsnip
    #winsorization and scaling
    SRE_train_scaled_winsorized <- SRE_train
    SRE_test_scaled_winsorized <- SRE_test
    
    winsorizable <- get_splineworthy_columns(SRE_train)
    
    #SRE_train_scaled_winsorized[lapply(SRE_train, n_distinct)>2] <- lapply((SRE_train[lapply(SRE_train, n_distinct)>2]), function(x) 0.4*(Winsorize(x, probs = c(0.025,0.975)))/sd(x))
    #for (col in names(SRE_train)[get_splineworthy_columns(SRE_train)]) {
    #  # Calculate winsorization limits for the current column in SRE_train
    #  winsor_limits <-  Winsorize(SRE_train[[col]], probs = c(0.025, 0.975))
    #  
    #  # Apply winsorization to the current column in both training and testing sets
    #  SRE_train_scaled_winsorized[[col]] <- Winsorize(SRE_train[[col]], minval = min(winsor_limits), maxval = max(winsor_limits))
    #  SRE_train_scaled_winsorized[[col]] <- 0.4*SRE_train_scaled_winsorized[[col]] / sd(SRE_train_scaled_winsorized[[col]])
    #  SRE_test_scaled_winsorized[[col]] <- 0.4*Winsorize(SRE_test[[col]], minval = min(winsor_limits), maxval = max(winsor_limits)) / sd(SRE_train_scaled_winsorized[[col]])
    #}
    SRE_recipe <- recipe(label~., data = SRE_train) %>%
      step_hai_winsorized_truncate(all_of(names(SRE_train)[winsorizable])&!contains("s("), fraction = 0.025) %>%
      step_rm(all_of(names(SRE_train)[winsorizable])&!contains("s(")) %>%
      step_mutate_at(contains("winsorized"), fn = ~0.4 * ./ sd(.)) %>%
      step_zv()
    
    SRE_train_baked <- SRE_recipe %>% prep() %>% bake(SRE_train)
    SRE_test_baked <- SRE_recipe %>% prep(SRE_train) %>% bake(SRE_test)
    
    
    
    
    
    
    
    
    #regular lasso
    SRE_model <- 
      parsnip::logistic_reg(
        mode = "classification",
        mixture = 1,
        penalty = tune()
      ) %>%
      set_engine("glmnet")
    
    SRE_wf <- workflow() %>%
      add_formula(label~.) %>%
      #add_recipe(LINEAR_recipe) %>%
      add_model(SRE_model)
    
    SRE_tuned <- tune::tune_grid(
      object = SRE_wf,
      resamples = inner_folds,
      grid = hyperparameters_SRE_tidy, 
      metrics = metrics,
      control = tune::control_grid(verbose = TRUE, save_pred = TRUE)
    )
    
    best_model_auc <- SRE_tuned %>% select_best("roc_auc")
    final_SRE_wf_auc <- SRE_wf %>% finalize_workflow(best_model_auc)
    final_SRE_fit_auc <- final_SRE_wf_auc %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    best_model_brier <- SRE_tuned %>% select_best("brier_class")
    final_SRE_wf_brier <- SRE_wf %>% finalize_workflow(best_model_brier)
    final_SRE_fit_brier <- final_SRE_wf_brier %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    best_model_pg <- SRE_tuned %>% select_best_pg_SRE()
    final_SRE_wf_pg <- SRE_wf %>% finalize_workflow(best_model_pg)
    final_SRE_fit_pg <- final_SRE_wf_pg %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    
    
    auc <- final_SRE_fit_auc %>%
      collect_metrics() %>%
      filter(.metric == "roc_auc") %>%
      pull(.estimate)
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE", auc)
    
    brier <- final_SRE_fit_brier %>%
      collect_metrics() %>%
      filter(.metric == "brier_class") %>%
      pull(.estimate)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE", brier)
    
    pg <- final_SRE_fit_pg %>%
      collect_pg()
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE", pg)
    
    
    
    
    #DEFINITIVE ADALASSO?
    # initial ridge coef
    set.seed(innerseed)
    ridge <- train(label ~., data = SRE_train_baked, #ERROR
                       method = "glmnet",
                       tuneGrid = expand.grid(alpha = 0, lambda = seq(0.001, 1, length = 100)),
                       metric = "AUCROC",
                       trControl = ctrl)
    lambda_auc <- ridge$results$lambda[which.max(ridge$results$AUCROC)]
    lambda_brier <- ridge$results$lambda[which.min(ridge$results$Brier)]
    lambda_pg <- ridge$results$lambda[which.max(ridge$results$partialGini)]
    
    set.seed(innerseed)
    ridge_auc <- train(label ~., data = SRE_train_baked, 
                       method = "glmnet",
                       tuneGrid = expand.grid(alpha = 0, lambda = lambda_auc),
                       trControl = trainControl(allowParallel = TRUE))    
    set.seed(innerseed)
    ridge_brier <- train(label ~., data = SRE_train_baked,
                       method = "glmnet",
                       tuneGrid = expand.grid(alpha = 0, lambda = lambda_brier),
                       trControl = trainControl(allowParallel = TRUE))    
    set.seed(innerseed)
    ridge_pg <- train(label ~., data = SRE_train_baked, 
                       method = "glmnet",
                       tuneGrid = expand.grid(alpha = 0, lambda = lambda_pg),
                       trControl = trainControl(allowParallel = TRUE))    
    
    coef_ridge_auc <- coef(ridge_auc$finalModel, s=lambda_auc)
    coef_ridge_brier <- coef(ridge_brier$finalModel, s=lambda_brier)
    coef_ridge_pg <- coef(ridge_pg$finalModel, s=lambda_pg)
    
    #final lasso
    set.seed(innerseed)
    adaLasso_auc <- train(label ~., data = SRE_train_baked, #ERROR
                          method = "glmnet",
                          tuneGrid = expand.grid(alpha = 1, lambda = seq(0.001, 1, length = 100)),
                          penalty.factor = 1/abs(coef_ridge_auc)[-1],
                          metric = "AUCROC",
                          trControl = ctrl)
    set.seed(innerseed)
    adaLasso_brier <- train(label ~., data = SRE_train_baked, #ERROR
                          method = "glmnet",
                          tuneGrid = expand.grid(alpha = 1, lambda = seq(0.001, 1, length = 100)),
                          penalty.factor = 1/abs(coef_ridge_brier)[-1],
                          metric = "Brier",
                          trControl = ctrl)
    set.seed(innerseed)
    adaLasso_pg <- train(label ~., data = SRE_train_baked, #ERROR
                          method = "glmnet",
                          tuneGrid = expand.grid(alpha = 1, lambda = seq(0.001, 1, length = 100)),
                          penalty.factor = 1/abs(coef_ridge_pg)[-1],
                          metric = "partialGini",
                          trControl = ctrl)
    
    
    preds_SRE_auc <- data.frame(predict(adaLasso_auc, SRE_test_baked, s = "lambda.min", type = "prob"))
    preds_SRE_auc$label <- SRE_test_baked$label
    #colnames(preds_SRE_auc) <- c("X1", "label")
    
    preds_SRE_brier <- data.frame(predict(adaLasso_brier, SRE_test_baked, s = "lambda.min", type = "prob"))
    preds_SRE_brier$label <- SRE_test_baked$label
    #colnames(preds_SRE_brier) <- c("X1", "label")
    
    preds_SRE_pg <- data.frame(predict(adaLasso_pg, SRE_test_baked, s = "lambda.min", type = "prob"))
    preds_SRE_pg$label <- SRE_test_baked$label
    #colnames(preds_SRE_pg) <- c("X1", "label")
    
    g <- roc(label ~ X1, data = preds_SRE_auc, direction = "<")
    AUC <- g$auc
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE", AUC)

    brier <- brier_class_vec(preds_SRE_brier$label, preds_SRE_brier$X1)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE", brier)
    
    pg <- partialGini(preds_SRE_pg$X1, preds_SRE_pg$label)
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE", pg)

  }
  write.csv(AUC_results, file = paste("./results/",dataset_vector[4],"_AUC.csv", sep = ""))
  write.csv(Brier_results, file = paste("./results/",dataset_vector[4],"_BRIER.csv", sep = ""))
  PG_results$metric<-unlist(PG_results$metric)
  write.csv(PG_results, file = paste("./results/",dataset_vector[4],"_PG.csv", sep = ""))
  
  dataset_counter <- dataset_counter + 1
}




#write.csv(metric_results, file = "./results/AUCROC_results.csv")

stopCluster(cl)


set.seed(123)
train_indices <- sample(1:nrow(x), 0.8 * nrow(x))
x_train <- (x[train_indices, ])
x_test <- (x[-train_indices, ])
y_train <- (y[train_indices, ])
y_test <- (y[-train_indices, ])
