# main
if (!require("pacman")) install.packages("pacman") ; require("pacman")
p_load(glmnet, glmnetUtils, mgcv, MASS, tidyverse, xgboost, DiagrammeR, stringr, tictoc, doParallel, pROC, earth, Matrix, pre, caret, parsnip, ggplot2, recipes, rsample, workflows, healthyR.ai, rlang, yardstick, bonsai, lightgbm, ranger, tune, DescTools, rules, discrim)
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
innerfolds = 5
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
  #MINIMAL_recipe <- recipe(label ~., data = dataset) %>%
  #  step_zv(all_predictors()) %>%
  #  step_dummy(all_string_predictors()) %>%
  #  step_dummy(all_factor_predictors())
  #
  #dataset <- MINIMAL_recipe %>% prep() %>% bake(dataset)                        #dit misschien trainen?

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
    
    
    
    #Needed for RF hyperparameters
    train_bake <- XGB_recipe %>% prep(train) %>% bake(train)
    test_bake <- XGB_recipe %>% prep(train) %>% bake(test)
    train_bake_x <- train_bake %>% dplyr::select(-label)

    
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
    
    GAM_recipe <- recipe(label~., data = train) %>%
      step_impute_mean(all_numeric_predictors()) %>%
      step_impute_mode(all_string_predictors()) %>%
      step_impute_mode(all_factor_predictors()) %>%
      #step_hai_winsorized_truncate(all_numeric_predictors(), fraction = 0.025) %>%
      #step_rm(!contains("winsorized") & all_numeric_predictors()) %>%
      step_dummy(all_string_predictors()) %>%
      step_dummy(all_factor_predictors()) %>%
      step_zv(all_predictors()) %>%
      step_normalize(all_numeric_predictors())
    
    train_processed <- GAM_recipe%>%prep()%>%bake(train)

    smooth_vars = colnames(train_processed%>%dplyr::select(-label))[get_splineworthy_columns(train_processed)]
    formula <- as.formula(
      stringr::str_sub(paste("label ~", 
                             paste(ifelse(names(train_processed%>%dplyr::select(-label)) %in% smooth_vars, "s(", ""),
                                   names(train_processed%>%dplyr::select(-label)),
                                   ifelse(names(train_processed%>%dplyr::select(-label)) %in% smooth_vars, ")",""),
                                   collapse = " + ")
      ), 0, -1)
    )
    
    GAM_model <- 
      parsnip::gen_additive_mod() %>%
        set_mode("classification") %>%
        set_engine("mgcv")
    
    GAM_wf <- workflow() %>%
      add_recipe(GAM_recipe) %>%
      add_model(GAM_model, formula = formula)
    
    final_GAM_fit <- GAM_wf %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    auc <- final_GAM_fit %>%
      collect_metrics() %>%
      filter(.metric == "roc_auc") %>%
      pull(.estimate)
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "GAM", auc)
    
    brier <- final_GAM_fit %>%
      collect_metrics() %>%
      filter(.metric == "brier_class") %>%
      pull(.estimate)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "GAM", brier)
    
    pg <- final_GAM_fit %>%
      collect_pg()
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "GAM", pg)
    
    
    
    #####
    # LDA #needs CFS step_corr tidy
    #####
    print("LDA")
  #step_corr doesnt work?
    corr_recipe <- recipe(label~., train) %>%
      step_impute_mean(all_numeric_predictors()) %>%
      step_impute_mode(all_string_predictors()) %>%
      step_impute_mode(all_factor_predictors()) %>%
      step_corr(all_numeric_predictors(), threshold = 0.8) %>%
      prep()
    train_bake_selected <- corr_recipe %>% bake(train)
    test_bake_selected <- corr_recipe %>% bake(test)

    LDA_model <- lda(label~., train_bake_selected)
    LDA_preds <- data.frame(predict(LDA_model, test_bake_selected, type = 'prob')$posterior)
    LDA_preds$label <- test_bake_selected$label
    #AUC
    g <- roc(label ~ X1, data = LDA_preds, direction = "<")
    AUC <- g$auc
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "LDA", AUC)

    brier <- brier_score(preds = LDA_preds$X1, truth = LDA_preds$label)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "LDA", brier)
    
    partialGini(actuals = LDA_preds$label, preds = LDA_preds$X1)
    
    #####
    # QDA #needs CFS
    #####
#    print("QDA")
#    
#    train_bake_selected_dummies <- XGB_recipe %>% prep(train_bake_selected) %>% bake(train_bake_selected)
#    test_bake_selected_dummies <- XGB_recipe %>% prep(train_bake_selected) %>% bake(test_bake_selected)
#    
#    QDA_model <- qda(x = train_bake_selected%>%dplyr::select(-label), grouping = train_bake_selected$label)
#    
#    #works but suboptimal
#    LDA_model <- lda(label~., data = train_bake)
#    QDA_preds <- data.frame(predict(object = QDA_model, newdata = test_bake_selected%>%dplyr::select(-label) %>%
#  mutate_if(is.character, as.numeric), type = 'prob')$posterior)
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
                               tuneGrid = expand.grid(mincriterion = (CTREE_model$results%>%slice_max(Brier)%>%dplyr::select(mincriterion))[[1]])) 
    CTREE_preds <- predict(CTREE_model_Brier, test, type = 'prob')
    CTREE_preds$label <- test$label
    
    
    brier <- brier_score(truth = CTREE_preds$label, preds = CTREE_preds$X1)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "CTREE", brier)
    
    #PG
    
    CTREE_model_PG <- train(TREE_recipe, data = train, method = "ctree",
                               tuneGrid = expand.grid(mincriterion = (CTREE_model$results%>%slice_max(partialGini)%>%dplyr::select(mincriterion))[[1]]))
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
    RF_preds_AUC <- data.frame(predict(RF_model_AUC, test, type = 'prob'))
    names(RF_preds_AUC) <- c("X0", "X1")
    RF_preds_AUC$label <- test$label
    g <- roc(label ~ X1, data = RF_preds_AUC, direction = "<")
    AUC <- g$auc
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "RF", AUC)
    
    #PG
    RF_model_PG <- extractBestModel(modellist, "partialGini")
    RF_preds_PG <- data.frame(predict(RF_model_PG, test, type = 'prob'))
    names(RF_preds_PG) <- c("X0", "X1")
    RF_preds_PG$label <- test$label
    pg <- partialGini(RF_preds_PG$X1, RF_preds_PG$label)
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "RF", pg)
    
    #Brier
    RF_model_Brier <- extractBestModel(modellist, "Brier")
    RF_preds_Brier <- data.frame(predict(RF_model_Brier, test, type = 'prob'))
    names(RF_preds_Brier) <- c("X0", "X1")
    RF_preds_Brier$label <- test$label
    bs <- brier_score(truth = RF_preds_Brier$label, preds = RF_preds_Brier$X1)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "RF", bs)
    
    
    #####
    # XGB
    #####
    print("XGB")

    set.seed(innerseed)
    
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

    RE_recipe <- train %>% recipe(label~.) %>% 
      step_impute_mean(all_numeric_predictors()) %>%
      step_impute_mode(all_string_predictors()) %>%
      step_impute_mode(all_factor_predictors()) %>%
      step_zv(all_predictors())
    
    set.seed(innerseed)
    RE_model <- train(XGB_recipe, data = train, method = "pre",
                      ntrees = min(500, round(nrow(train)/2)), family = "binomial", trControl = ctrl,
                      tuneGrid = preGrid, ad.alpha = 0, singleconditions = TRUE,
                      winsfrac = 0.05, normalize = TRUE, #same a priori influence as a typical rule
                      verbose = TRUE,
                      metric = "AUCROC", allowParallel = TRUE,
                      par.init=TRUE,
                      par.final=TRUE)    
    
    #AUC
    RE_preds <- predict(RE_model, test, type = 'probs')
    RE_preds$label <- test$label
    
    g <- roc(label ~ X1, data = RE_preds, direction = "<")
    AUC <- g$auc
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "RE", AUC)
    
    #Brier
    RE_model_Brier <- train(XGB_recipe, data = train, method = "pre",
                            ntrees = min(500, round(nrow(train)/2)), family = "binomial", trControl = trainControl(method = "none", classProbs = TRUE),
                            tuneGrid = getModelInfo("pre")[[1]]$grid( 
                              maxdepth = (RE_model$results%>%slice_max(Brier)%>%dplyr::select(maxdepth))[[1]],
                              learnrate = (RE_model$results%>%slice_max(Brier)%>%dplyr::select(learnrate))[[1]],
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
    brier <- brier_score(truth = RE_preds$label, preds = RE_preds$X1)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "RE", brier)
    
    #PG
    RE_model_PG <- train(XGB_recipe, data = train, method = "pre",
                            ntrees = min(500, round(nrow(train)/2)), family = "binomial", trControl = trainControl(method = "none", classProbs = TRUE),
                            tuneGrid = getModelInfo("pre")[[1]]$grid( 
                              maxdepth = (RE_model$results%>%slice_max(partialGini)%>%dplyr::select(maxdepth))[[1]],
                              learnrate = (RE_model$results%>%slice_max(partialGini)%>%dplyr::select(learnrate))[[1]],
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
    smooth_terms <- grep("s\\(", unlist(str_split(as.character(formula), " \\+ ")), value = TRUE)
    # Extract and fitted values for each smooth term
    fitted_smooths_train <- data.frame(matrix(ncol = length(smooth_terms), nrow = nrow(train_bake)))
    fitted_smooths_test <- data.frame(matrix(ncol = length(smooth_terms), nrow = nrow(test_bake)))
    colnames(fitted_smooths_train) <- smooth_terms
    colnames(fitted_smooths_test) <- smooth_terms
    for (j in seq_along(smooth_terms)) {
      current_smooth <- smooth_terms[j]
      fitted_values_train <- predict(extract_fit_engine(final_GAM_fit),train_bake, type = "terms")[, current_smooth]
      fitted_smooths_train[, j] <- fitted_values_train
      fitted_values_test <- predict(extract_fit_engine(final_GAM_fit), test_bake, type = "terms")[, current_smooth]
      fitted_smooths_test[, j] <- fitted_values_test 
    }
    
    SRE_recipe <- recipe(label ~., data = train) %>%
      #step_impute_mean(all_numeric_predictors()) %>%
      #step_impute_mode(all_string_predictors()) %>%
      #step_impute_mode(all_factor_predictors()) %>%
      step_dummy(all_string_predictors()) %>%
      step_dummy(all_factor_predictors())
    
    
    train_rule_baked <- XGB_recipe %>%
      prep(train) %>%
      bake(train)
    test_rule_baked <- XGB_recipe %>%
      prep(train) %>%
      bake(test)    
    SRE_train_rules <- fit_rules(train_rule_baked, drop_na(tibble(rules = RE_model$finalModel$rules$description))$rules)
    SRE_test_rules <- fit_rules(test_rule_baked, drop_na(tibble(rules = RE_model$finalModel$rules$description))$rules)
    
    SRE_train <- cbind(SRE_train_rules, fitted_smooths_train)
    SRE_test <- cbind(SRE_test_rules, fitted_smooths_test)
    

    winsorizable <- get_splineworthy_columns(SRE_train)
 
    
    indices <- list(
      list(analysis = 1:nrow(SRE_train), assessment = (nrow(SRE_train)+1):(nrow(SRE_train)+nrow(SRE_test)))
    )
    
    splits <- lapply(indices, make_splits, data = rbind(SRE_train, SRE_test))
    
    SRE_split <- manual_rset(splits, c("Split SRE"))
    
    normalizable <- colnames(training(SRE_split$splits[[1]])[unlist(lapply(training(SRE_split$splits[[1]]), function(x) n_distinct(x)>2))])
    SRE_recipe <- recipe(label~., data = training(SRE_split$splits[[1]])) %>%
      step_hai_winsorized_truncate(all_of(names(!!training(SRE_split$splits[[1]]))[!!winsorizable]), fraction = 0.025) %>%
      step_rm(all_of(names(!!training(SRE_split$splits[[1]]))[!!winsorizable])) %>%
      step_mutate_at(contains("winsorized"), fn = ~0.4 * ./ sd(.)) %>%
      step_mutate(across(where(is.logical), as.integer)) %>%
      step_normalize(all_of(setdiff(!!normalizable, colnames(!!training(SRE_split$splits[[1]])[!!winsorizable])))) %>%
      step_zv()
    
    set.seed(i)
    inner_folds_SRE <- SRE_train %>% vfold_cv(v=5)
    
    #regular lasso
    SRE_model <- 
      parsnip::logistic_reg(
        mode = "classification",
        mixture = 1,
        penalty = tune()
      ) %>%
      set_engine("glmnet")
    
    SRE_wf <- workflow() %>%
      #add_formula(label~.) %>%
      add_recipe(SRE_recipe) %>%
      add_model(SRE_model)
    
    SRE_tuned <- tune::tune_grid(
      object = SRE_wf,
      resamples = inner_folds_SRE,
      grid = hyperparameters_SRE_tidy, 
      metrics = metrics,
      control = tune::control_grid(verbose = TRUE, save_pred = TRUE)
    )
    

    SRE_metrics <- SRE_tuned %>% 
      collect_metrics()
    
    lambda_min_auc <- SRE_metrics %>%
      filter(.metric=="roc_auc") %>%
      slice_max(mean) %>%
      dplyr::select(penalty) %>%
      pull()
    
    lambda_min_brier <- SRE_metrics %>%
      filter(.metric=="brier_class") %>%
      slice_min(mean) %>%
      dplyr::select(penalty) %>%
      pull()
    
    lambda_min_pg <- suppressMessages(SRE_tuned %>%
      collect_predictions(summarize = FALSE) %>%
      group_by(id, penalty, .config) %>%
      summarise(partial_gini = partialGini(.pred_X1, label)) %>%
      group_by(penalty, .config) %>%
      summarise(avg_pg = mean(partial_gini)) %>%
      ungroup() %>%
      slice_max(avg_pg) %>%
      slice_head() %>%
      dplyr::select(penalty) %>%                            
      pull())
    

    lambda_sd_auc <- sd(unlist(lapply(SRE_tuned$.metrics, function(tbl) {
      # Filter to keep rows where penalty is "0.0111"
      # Note: Ensure the penalty column is treated as character if it's not numeric
      tbl %>% filter(.metric=="roc_auc") %>%
        slice_max(.estimate) %>%
        dplyr::select(penalty)    
      })))
    
    lambda_sd_brier <- sd(unlist(lapply(SRE_tuned$.metrics, function(tbl) {
      # Filter to keep rows where penalty is "0.0111"
      # Note: Ensure the penalty column is treated as character if it's not numeric
      tbl %>% filter(.metric=="brier_class") %>%
        slice_min(.estimate) %>%
        dplyr::select(penalty)    
    })))
    
    lambda_sd_pg <- suppressMessages(SRE_tuned %>%
      collect_predictions(summarize = FALSE) %>%
      group_by(id, penalty, .config) %>%
      summarise(partial_gini = partialGini(.pred_X1, label)) %>%
      group_by(id) %>%
      slice_max(partial_gini) %>%
      ungroup() %>%
      summarise(sd(penalty)) %>%
      pull())
    
    lambda_1se_auc <- lambda_min_auc + lambda_sd_auc
    lambda_1se_brier <- lambda_min_brier + lambda_sd_brier
    lambda_1se_pg <- lambda_min_pg + lambda_sd_pg
    
  
    SRE_model_auc <- 
      parsnip::logistic_reg(
        mode = "classification",
        mixture = 1,
        penalty = lambda_1se_auc
      ) %>%
      set_engine("glmnet")
    
    SRE_wf_auc <- workflow() %>%
      #add_formula(label~.) %>%
      add_recipe(SRE_recipe) %>%
      add_model(SRE_model_auc)
    
    SRE_model_brier <- 
      parsnip::logistic_reg(
        mode = "classification",
        mixture = 1,
        penalty = lambda_1se_brier
      ) %>%
      set_engine("glmnet")
    
    SRE_wf_brier <- workflow() %>%
      #add_formula(label~.) %>%
      add_recipe(SRE_recipe) %>%
      add_model(SRE_model_brier)    
    
    SRE_model_pg <- 
      parsnip::logistic_reg(
        mode = "classification",
        mixture = 1,
        penalty = lambda_1se_pg
      ) %>%
      set_engine("glmnet")
    
    SRE_wf_pg <- workflow() %>%
      #add_formula(label~.) %>%
      add_recipe(SRE_recipe) %>%
      add_model(SRE_model_pg)    

    final_SRE_fit_auc <- SRE_wf_auc %>% last_fit(SRE_split$splits[[1]], metrics = metrics)
    
    final_SRE_fit_brier <- SRE_wf_brier %>% last_fit(SRE_split$splits[[1]], metrics = metrics)
    
    final_SRE_fit_pg <- SRE_wf_pg %>% last_fit(SRE_split$splits[[1]], metrics = metrics)
    
    
    
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
    
  
  }
  write.csv(AUC_results, file = paste("./results/",dataset_vector[1],"_AUC.csv", sep = ""))
  write.csv(Brier_results, file = paste("./results/",dataset_vector[1],"_BRIER.csv", sep = ""))
  PG_results$metric<-unlist(PG_results$metric)
  write.csv(PG_results, file = paste("./results/",dataset_vector[1],"_PG.csv", sep = ""))
  
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
