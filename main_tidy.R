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
nr_repeats = 5
outerfolds = 2
nr_innerfolds = 5
dataset_vector = c("GC", "AC", "HMEQ", "TH02", "LC", "TC", "GMSC")

ctrl <- trainControl(method = "cv", number = nr_innerfolds, classProbs = TRUE, summaryFunction = BigSummary, search = "grid", allowParallel = TRUE)
metrics = metric_set(roc_auc, brier_class)

# create empty dataframe metric_results with columns: (dataset, repeat, fold, algorithm, metric)	
metric_results <- data.frame(
  dataset = character(),
  nr_fold = integer(),
  algorithm = character(),
  metric = double(),
  stringsAsFactors = FALSE
)

dataset_counter = 1

for(dataset in datasets) {
  
  AUC_results <- metric_results
  Brier_results <- metric_results
  PG_results <- metric_results
  
  nr_innerfolds = nr_repeats
  
  set.seed(111)
  # create 5x2 folds
  folds <- vfold_cv(dataset, v = outerfolds, repeats = nr_repeats, strata = NULL)
  for(i in 1:nrow(folds)) { #CHANGE
    cat("Fold", i, "/ 10 \n")
    train <- analysis(folds$splits[[i]])
    test <- assessment(folds$splits[[i]])
    
    # 
    # for tree-based that don't require dummies
    TREE_recipe <- recipe(label ~., data = train) %>%
      step_impute_mean(all_numeric_predictors()) %>%
      step_impute_mode(all_string_predictors()) %>%
      step_impute_mode(all_factor_predictors()) %>%
      step_novel(all_nominal_predictors()) %>%
      step_zv(all_predictors())
    
    # for tree-based that do require dummies
    XGB_recipe <- recipe(label ~., data = train) %>%
      step_impute_mean(all_numeric_predictors()) %>%
      step_impute_mode(all_string_predictors()) %>%
      step_impute_mode(all_factor_predictors()) %>%
      step_novel(all_nominal_predictors()) %>%
      step_dummy(all_string_predictors()) %>%
      step_dummy(all_factor_predictors()) %>%
      step_zv(all_predictors())
    
    winsorizable <- names(train)[get_splineworthy_columns(train)]
    
    # for algorithms using linear terms (LRR, gam, rule ensembles)
    LINEAR_recipe <- recipe(label ~., data = train) %>%
      step_impute_mean(all_numeric_predictors()) %>%
      step_impute_mode(all_string_predictors()) %>%
      step_impute_mode(all_factor_predictors()) %>%
      step_novel(all_nominal_predictors()) %>%
      step_hai_winsorized_truncate(all_numeric_predictors(), fraction = 0.025) %>%
      step_rm(!contains("winsorized") & all_numeric_predictors()) %>%
      step_dummy(all_string_predictors()) %>%
      step_dummy(all_factor_predictors()) %>%
      step_zv(all_predictors()) %>%
      step_normalize(all_numeric_predictors())
    
    LDA_recipe <- recipe(label~., train) %>%
      step_impute_mean(all_numeric_predictors()) %>%
      step_impute_mode(all_string_predictors()) %>%
      step_impute_mode(all_factor_predictors()) %>%
      step_novel(all_nominal_predictors()) %>%
      step_dummy(all_string_predictors()) %>%
      step_dummy(all_factor_predictors()) %>%
      step_corr(all_numeric_predictors(), threshold = 0.8) %>%
      step_zv(all_predictors())
    
    GAM_recipe <- recipe(label~., data = train) %>%
      step_impute_mean(all_numeric_predictors()) %>%
      step_impute_mode(all_string_predictors()) %>%
      step_impute_mode(all_factor_predictors()) %>%
      #step_hai_winsorized_truncate(all_numeric_predictors(), fraction = 0.025) %>%
      #step_rm(!contains("winsorized") & all_numeric_predictors()) %>%
      step_novel(all_nominal_predictors()) %>%
      step_dummy(all_string_predictors()) %>%
      step_dummy(all_factor_predictors()) %>%
      step_zv(all_predictors()) %>%
      step_normalize(all_numeric_predictors())
    
    
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
      
    train_bake_selected <- LDA_recipe %>% prep(train) %>% bake(train)
    test_bake_selected <- LDA_recipe %>% prep(train) %>% bake(test)

    LDA_model <- lda(label~., train_bake_selected)
    LDA_preds <- data.frame(predict(LDA_model, test_bake_selected, type = 'prob')$posterior)
    LDA_preds$label <- test_bake_selected$label
    #AUC
    g <- roc(label ~ X1, data = LDA_preds, direction = "<")
    AUC <- g$auc
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "LDA", AUC)

    brier <- brier_score(preds = LDA_preds$X1, truth = LDA_preds$label)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "LDA", brier)
    
    pg <- partialGini(actuals = LDA_preds$label, preds = LDA_preds$X1)    
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "LDA", pg)

    
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
    
    
    #tidy
    
    RF_model <- 
      parsnip::rand_forest(
        mode = "classification",
        trees = tune(),
        mtry = tune(),
        min_n = tune()
      ) %>%
      set_engine("ranger")
    
    RF_wf <- workflow() %>%
      add_recipe(XGB_recipe) %>%
      add_model(RF_model)
    
    RF_tuned <- tune::tune_grid(
      object = RF_wf,
      resamples = inner_folds,
      grid = hyperparameters_RF_tidy,
      metrics = metrics,
      control = tune::control_grid(verbose = TRUE, save_pred = TRUE)
    )
    
    
    best_model_auc <- RF_tuned %>% select_best("roc_auc")
    final_RF_wf_auc <- RF_wf %>% finalize_workflow(best_model_auc)
    final_RF_fit_auc <- final_RF_wf_auc %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    best_model_brier <- RF_tuned %>% select_best("brier_class")
    final_RF_wf_brier <- RF_wf %>% finalize_workflow(best_model_brier)
    final_RF_fit_brier <- final_RF_wf_brier %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    best_model_pg <- RF_tuned %>% select_best_pg_RF()
    final_RF_wf_pg <- RF_wf %>% finalize_workflow(best_model_pg)
    final_RF_fit_pg <- final_RF_wf_pg %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    
    auc <- final_RF_fit_auc %>%
      collect_metrics() %>%
      filter(.metric == "roc_auc") %>%
      pull(.estimate)
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "RF", auc)
    
    brier <- final_RF_fit_brier %>%
      collect_metrics() %>%
      filter(.metric == "brier_class") %>%
      pull(.estimate)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "RF", brier)
    
    pg <- final_RF_fit_pg %>%
      collect_pg()
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "RF", pg)
    
    
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
        sample_size = tune(),
        mtry = tune()
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
        mtry = tune()
      ) %>%
      set_engine("lightgbm") %>%
      translate()
    
    lgbm_wf <- workflow() %>%
      add_recipe(XGB_recipe) %>%
      add_model(lgbm_model)
    
    lgbm_tuned <- tune::tune_grid(
      object = lgbm_wf,
      resamples = inner_folds, 
      grid = hyperparameters_LGBM_tidy, 
      metrics = metrics,
      control = tune::control_grid(verbose = TRUE, save_pred = TRUE)
    )
    
    best_booster_auc <- lgbm_tuned %>% select_best("roc_auc")
    final_lgbm_wf_auc <- lgbm_wf %>% finalize_workflow(best_booster_auc)
    final_lgbm_fit_auc <- final_lgbm_wf_auc %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    best_booster_brier <- lgbm_tuned %>% select_best("brier_class")
    final_lgbm_wf_brier <- lgbm_wf %>% finalize_workflow(best_booster_brier)
    final_lgbm_fit_brier <- final_lgbm_wf_brier %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    best_model_pg <- lgbm_tuned %>% select_best_pg_LGBM()
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
    
    pg <- final_lgbm_fit_pg %>%
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
      step_novel(all_nominal_predictors()) %>%
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
                            ntrees = min(100, round(nrow(train)/2)), family = "binomial", trControl = trainControl(method = "none", classProbs = TRUE),
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
                            ntrees = min(100, round(nrow(train)/2)), family = "binomial", trControl = trainControl(method = "none", classProbs = TRUE),
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
    
    
    ####### 
    # inner SRE loop for hyperparameter tuning (glmnet)
    
    full_metrics_AUC <- list()
    full_metrics_Brier <- list()
    full_metrics_PG <- list()
    
    for(k in 1:nrow(inner_folds)) {
      cat("SRE inner fold", k, "/ 5 \n")
      ####### 
      # Data is split in training and test, preprocessing is applied
      
      inner_train <- analysis(inner_folds$splits[[k]])
      inner_test <- assessment(inner_folds$splits[[k]])
      
      inner_train_bake <- XGB_recipe %>% prep(inner_train) %>% bake(inner_train)
      inner_test_bake <- XGB_recipe %>% prep(inner_train) %>% bake(inner_test)
      
      ####### 
      # Fit GAM to extract splines only on numeric features with number of values >6
      
      inner_train_gam_processed <- GAM_recipe%>%prep(inner_train)%>%bake(inner_train)
      
      smooth_vars = colnames(inner_train_gam_processed%>%dplyr::select(-label))[get_splineworthy_columns(inner_train_gam_processed)]
      formula <- as.formula(
        stringr::str_sub(paste("label ~", 
                               paste(ifelse(names(inner_train_gam_processed%>%dplyr::select(-label)) %in% smooth_vars, "s(", ""),
                                     names(inner_train_gam_processed%>%dplyr::select(-label)),
                                     ifelse(names(inner_train_gam_processed%>%dplyr::select(-label)) %in% smooth_vars, ")",""),
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
      
      final_GAM_fit_inner <- GAM_wf %>% last_fit(folds$splits[[i]], metrics = metrics)
      
      
      # Extract and fitted values for each smooth term
      smooth_terms <- grep("s\\(", unlist(str_split(as.character(formula), " \\+ ")), value = TRUE)
      fitted_smooths_train <- data.frame(matrix(ncol = length(smooth_terms), nrow = nrow(inner_train_bake)))
      fitted_smooths_test <- data.frame(matrix(ncol = length(smooth_terms), nrow = nrow(inner_test_bake)))
      colnames(fitted_smooths_train) <- smooth_terms
      colnames(fitted_smooths_test) <- smooth_terms
      for (j in seq_along(smooth_terms)) {
        current_smooth <- smooth_terms[j]
        fitted_values_train <- predict(extract_fit_engine(final_GAM_fit_inner), inner_train_bake, type = "terms")[, current_smooth]
        fitted_smooths_train[, j] <- fitted_values_train
        fitted_values_test <- predict(extract_fit_engine(final_GAM_fit_inner), inner_test_bake, type = "terms")[, current_smooth]
        fitted_smooths_test[, j] <- fitted_values_test 
      }
      
      ####### 
      # Fit rules from RE_models, seperate for AUC, Brier, PG
      
      if(!is.null(RE_model$finalModel$rules)) {
        SRE_train_rules_AUC <- fit_rules(inner_train_bake, drop_na(tibble(rules = RE_model$finalModel$rules$description))$rules)
        SRE_test_rules_AUC <- fit_rules(inner_test_bake, drop_na(tibble(rules = RE_model$finalModel$rules$description))$rules)
        
        SRE_train_AUC <- cbind(SRE_train_rules_AUC, fitted_smooths_train)
        SRE_test_AUC <- cbind(SRE_test_rules_AUC, fitted_smooths_test)
      } else {
        SRE_train_AUC <- cbind(inner_train_bake, fitted_smooths_train)
        SRE_test_AUC <- cbind(inner_test_bake, fitted_smooths_test)
      }
      if(!is.null(RE_model_Brier$finalModel$rules)) {
        SRE_train_rules_Brier <- fit_rules(inner_train_bake, drop_na(tibble(rules = RE_model_Brier$finalModel$rules$description))$rules)
        SRE_test_rules_Brier<- fit_rules(inner_test_bake, drop_na(tibble(rules = RE_model_Brier$finalModel$rules$description))$rules)
        
        SRE_train_Brier <- cbind(SRE_train_rules_Brier, fitted_smooths_train)
        SRE_test_Brier <- cbind(SRE_test_rules_Brier, fitted_smooths_test)
      } else {
        SRE_train_Brier <- cbind(inner_train_bake, fitted_smooths_train)
        SRE_test_Brier <- cbind(inner_test_bake, fitted_smooths_test)
      }
      if(!is.null(RE_model_PG$finalModel$rules)) {
        SRE_train_rules_PG <- fit_rules(inner_train_bake, drop_na(tibble(rules = RE_model_PG$finalModel$rules$description))$rules)
        SRE_test_rules_PG <- fit_rules(inner_test_bake, drop_na(tibble(rules = RE_model_PG$finalModel$rules$description))$rules)
        
        SRE_train_PG <- cbind(SRE_train_rules_PG, fitted_smooths_train)
        SRE_test_PG <- cbind(SRE_test_rules_PG, fitted_smooths_test)
      } else {
        SRE_train_PG <- cbind(inner_train_bake, fitted_smooths_train)
        SRE_test_PG <- cbind(inner_test_bake, fitted_smooths_test)
      }
      
      ####### 
      # Again, only winsorize numeric features with more than 6 values
      winsorizable_AUC <- get_splineworthy_columns(SRE_train_AUC)
      winsorizable_Brier <- get_splineworthy_columns(SRE_train_Brier)
      winsorizable_PG <- get_splineworthy_columns(SRE_train_PG)
      
      #######
      # Manually create Rsample splits again, now with the splines and rules added
      indices <- list(
        list(analysis = 1:nrow(SRE_train_AUC), assessment = (nrow(SRE_train_AUC)+1):(nrow(SRE_train_AUC)+nrow(SRE_test_AUC)))
      )
      splits_AUC <- lapply(indices, make_splits, data = rbind(SRE_train_AUC, SRE_test_AUC))
      splits_Brier <- lapply(indices, make_splits, data = rbind(SRE_train_Brier, SRE_test_Brier))
      splits_PG <- lapply(indices, make_splits, data = rbind(SRE_train_PG, SRE_test_PG))
      SRE_split_AUC <- manual_rset(splits_AUC, c("Split SRE"))
      SRE_split_Brier <- manual_rset(splits_Brier, c("Split SRE"))
      SRE_split_PG <- manual_rset(splits_PG, c("Split SRE"))
      
      ####### 
      # Create recipes, for AUC, Brier and PG
      normalizable_AUC <- colnames(training(SRE_split_AUC$splits[[1]])[unlist(lapply(training(SRE_split_AUC$splits[[1]]), function(x) n_distinct(x)>2))])
      normalizable_Brier <- colnames(training(SRE_split_Brier$splits[[1]])[unlist(lapply(training(SRE_split_Brier$splits[[1]]), function(x) n_distinct(x)>2))])
      normalizable_PG <- colnames(training(SRE_split_PG$splits[[1]])[unlist(lapply(training(SRE_split_PG$splits[[1]]), function(x) n_distinct(x)>2))])

      SRE_recipe_AUC <- recipe(label~., data = training(SRE_split_AUC$splits[[1]])) %>%
        step_hai_winsorized_truncate(all_of(names(!!training(SRE_split_AUC$splits[[1]]))[!!winsorizable_AUC]), fraction = 0.025) %>%
        step_rm(all_of(names(!!training(SRE_split_AUC$splits[[1]]))[!!winsorizable_AUC])) %>%
        step_mutate_at(contains("winsorized"), fn = ~0.4 * ./ sd(.)) %>%
        step_mutate(across(where(is.logical), as.integer)) %>%
        step_normalize(all_of(setdiff(!!normalizable_AUC, colnames(!!training(SRE_split_AUC$splits[[1]])[!!winsorizable_AUC])))) %>%
        step_zv()

      SRE_recipe_Brier <- recipe(label~., data = training(SRE_split_Brier$splits[[1]])) %>%
        step_hai_winsorized_truncate(all_of(names(!!training(SRE_split_Brier$splits[[1]]))[!!winsorizable_Brier]), fraction = 0.025) %>%
        step_rm(all_of(names(!!training(SRE_split_Brier$splits[[1]]))[!!winsorizable_Brier])) %>%
        step_mutate_at(contains("winsorized"), fn = ~0.4 * ./ sd(.)) %>%
        step_mutate(across(where(is.logical), as.integer)) %>%
        step_normalize(all_of(setdiff(!!normalizable_Brier, colnames(!!training(SRE_split_Brier$splits[[1]])[!!winsorizable_Brier])))) %>%
        step_zv()

      SRE_recipe_PG <- recipe(label~., data = training(SRE_split_PG$splits[[1]])) %>%
        step_hai_winsorized_truncate(all_of(names(!!training(SRE_split_PG$splits[[1]]))[!!winsorizable_PG]), fraction = 0.025) %>%
        step_rm(all_of(names(!!training(SRE_split_PG$splits[[1]]))[!!winsorizable_PG])) %>%
        step_mutate_at(contains("winsorized"), fn = ~0.4 * ./ sd(.)) %>%
        step_mutate(across(where(is.logical), as.integer)) %>%
        step_normalize(all_of(setdiff(!!normalizable_PG, colnames(!!training(SRE_split_PG$splits[[1]])[!!winsorizable_PG])))) %>%
        step_zv()
      
      ####### 
      # Fit regular lasso for AUC, Brier, PG
      SRE_model <- 
        parsnip::logistic_reg(
          mode = "classification",
          mixture = 1,
          penalty = tune()
        ) %>%
        set_engine("glmnet")
      
      SRE_wf_AUC <- workflow() %>%
        add_recipe(SRE_recipe_AUC) %>%
        add_model(SRE_model)
      SRE_wf_Brier <- workflow() %>%
        add_recipe(SRE_recipe_Brier) %>%
        add_model(SRE_model)
      SRE_wf_PG <- workflow() %>%
        add_recipe(SRE_recipe_PG) %>%
        add_model(SRE_model)
      
      
      SRE_tuned_AUC <- tune::tune_grid(
        object = SRE_wf_AUC,
        resamples = SRE_split_AUC,
        grid = hyperparameters_SRE_tidy, 
        metrics = metrics,
        control = tune::control_grid(verbose = TRUE, save_pred = TRUE))
      SRE_tuned_Brier <- tune::tune_grid(
        object = SRE_wf_Brier,
        resamples = SRE_split_Brier,
        grid = hyperparameters_SRE_tidy, 
        metrics = metrics,
        control = tune::control_grid(verbose = TRUE, save_pred = TRUE))
      SRE_tuned_PG <- tune::tune_grid(
        object = SRE_wf_PG,
        resamples = SRE_split_PG,
        grid = hyperparameters_SRE_tidy, 
        metrics = metrics,
        control = tune::control_grid(verbose = TRUE, save_pred = TRUE))
      
      #for auc, brier
      metrics_SRE_AUC <- SRE_tuned_AUC$.metrics[[1]]
      metrics_SRE_Brier <- SRE_tuned_Brier$.metrics[[1]]
      metrics_SRE_AUC$fold <- rep(k, nrow(metrics_SRE_AUC))
      metrics_SRE_Brier$fold <- rep(k, nrow(metrics_SRE_Brier))

      full_metrics_AUC <- rbind(full_metrics_AUC, metrics_SRE_AUC)
      full_metrics_Brier <- rbind(full_metrics_Brier, metrics_SRE_Brier)

      #for pg
      metrics_SRE_PG <- suppressMessages(SRE_tuned_PG%>%collect_predictions(summarize = FALSE) %>%
        group_by(id, penalty, .config) %>%
        summarise(partial_gini = partialGini(.pred_X1, label)))
      metrics_SRE_PG$fold <- rep(k, nrow(metrics_SRE_PG))
      
      full_metrics_PG <- rbind(full_metrics_PG, metrics_SRE_PG)
    }
    
    print("hyperparameters found")
    
    ####### 
    # Hyperparameter extraction, we use lambda.1se  = lambda.min + 1se
    aggregated_metrics_AUC <- full_metrics_AUC %>% group_by(penalty, .config, .metric) %>%
      summarise(mean_perf = mean(.estimate))
    aggregated_metrics_Brier <- full_metrics_Brier %>% group_by(penalty, .config, .metric) %>%
      summarise(mean_perf = mean(.estimate))
    aggregated_metrics_PG <- full_metrics_PG %>% group_by(penalty, .config) %>%
      summarise(mean_perf = mean(partial_gini))
    
    best_lambda_auc <- aggregated_metrics_AUC %>% filter(.metric=="roc_auc") %>% ungroup() %>% slice_max(mean_perf) %>% slice_head() %>% dplyr::select(penalty) %>% pull()
    best_lambda_brier <- aggregated_metrics_Brier %>% filter(.metric=="brier_class") %>% ungroup() %>% slice_min(mean_perf) %>% slice_head() %>% dplyr::select(penalty) %>% pull()
    best_lambda_pg <- aggregated_metrics_PG %>% ungroup() %>% slice_max(mean_perf) %>% slice_head() %>% dplyr::select(penalty) %>% pull()
    
    lambda_sd_auc <- sd(unlist(
      full_metrics_AUC %>% filter(.metric=="roc_auc") %>%
        group_by(fold) %>%
        slice_max(.estimate) %>%
        slice_head() %>%
        ungroup() %>%
        dplyr::select(penalty)    
    ))
    
    lambda_sd_brier <- sd(unlist(
      full_metrics_Brier %>% filter(.metric=="brier_class") %>%
        group_by(fold) %>%
        slice_min(.estimate) %>%
        slice_head() %>%
        ungroup() %>%
        dplyr::select(penalty)    
    ))
    
    lambda_sd_pg <- sd(unlist(
      full_metrics_PG %>% 
        group_by(fold) %>%
        slice_max(partial_gini) %>%
        slice_head() %>%
        ungroup() %>%
        dplyr::select(penalty)    
    ))
    
    
    lambda_1se_auc <- best_lambda_auc + lambda_sd_auc
    lambda_1se_brier <- best_lambda_brier + lambda_sd_brier
    lambda_1se_pg <- best_lambda_pg + lambda_sd_pg
    

    ####### 
    # Preprocessing on full training and test set
    # Extract fitted values for each smooth term
    smooth_terms <- grep("s\\(", unlist(str_split(as.character(formula), " \\+ ")), value = TRUE)
    fitted_smooths_train <- data.frame(matrix(ncol = length(smooth_terms), nrow = nrow(train_bake)))
    fitted_smooths_test <- data.frame(matrix(ncol = length(smooth_terms), nrow = nrow(test_bake)))
    colnames(fitted_smooths_train) <- smooth_terms
    colnames(fitted_smooths_test) <- smooth_terms
    for (j in seq_along(smooth_terms)) {
      current_smooth <- smooth_terms[j]
      fitted_values_train <- predict(extract_fit_engine(final_GAM_fit), train_bake, type = "terms")[, current_smooth]
      fitted_smooths_train[, j] <- fitted_values_train
      fitted_values_test <- predict(extract_fit_engine(final_GAM_fit), test_bake, type = "terms")[, current_smooth]
      fitted_smooths_test[, j] <- fitted_values_test 
    }
    
    ####### 
    # Fit rules from RE_models, seperate for AUC, Brier, PG
    if(!is.null(RE_model$finalModel$rules)) {
      SRE_train_rules_AUC <- fit_rules(train_bake, drop_na(tibble(rules = RE_model$finalModel$rules$description))$rules)
      SRE_test_rules_AUC <- fit_rules(test_bake, drop_na(tibble(rules = RE_model$finalModel$rules$description))$rules)
    } else {
      SRE_train_rules_AUC <- train_bake
      SRE_test_rules_AUC <- test_bake
    }

    if(!is.null(RE_model_Brier$finalModel$rules)) {
      SRE_train_rules_Brier <- fit_rules(train_bake, drop_na(tibble(rules = RE_model_Brier$finalModel$rules$description))$rules)
      SRE_test_rules_Brier <- fit_rules(test_bake, drop_na(tibble(rules = RE_model_Brier$finalModel$rules$description))$rules)
    } else {
      SRE_train_rules_Brier <- train_bake
      SRE_test_rules_Brier <- test_bake
    }
    
    if(!is.null(RE_model_PG$finalModel$rules)) {
      SRE_train_rules_PG <- fit_rules(train_bake, drop_na(tibble(rules = RE_model_PG$finalModel$rules$description))$rules)
      SRE_test_rules_PG <- fit_rules(test_bake, drop_na(tibble(rules = RE_model_PG$finalModel$rules$description))$rules)
    } else {
      SRE_train_rules_PG <- train_bake
      SRE_test_rules_PG <- test_bake
    }
    
    
    SRE_train_AUC <- cbind(SRE_train_rules_AUC, fitted_smooths_train)
    SRE_train_Brier <- cbind(SRE_train_rules_Brier, fitted_smooths_train)
    SRE_train_PG <- cbind(SRE_train_rules_PG, fitted_smooths_train)
    
    SRE_test_AUC <- cbind(SRE_test_rules_AUC, fitted_smooths_test)
    SRE_test_Brier <- cbind(SRE_test_rules_Brier, fitted_smooths_test)
    SRE_test_PG <- cbind(SRE_test_rules_PG, fitted_smooths_test)
    
    ####### 
    # Only winsorize numeric features with more than 6 values
    winsorizable_AUC <- get_splineworthy_columns(SRE_train_AUC)
    winsorizable_Brier <- get_splineworthy_columns(SRE_train_Brier)
    winsorizable_PG <- get_splineworthy_columns(SRE_train_PG)
    
    #######
    # Manually create Rsample splits again, now with the splines and rules added
    indices <- list(
      list(analysis = 1:nrow(SRE_train_AUC), assessment = (nrow(SRE_train_AUC)+1):(nrow(SRE_train_AUC)+nrow(SRE_test_AUC)))
    )
    
    splits_AUC <- lapply(indices, make_splits, data = rbind(SRE_train_AUC, SRE_test_AUC))
    splits_Brier <- lapply(indices, make_splits, data = rbind(SRE_train_Brier, SRE_test_Brier))
    splits_PG <- lapply(indices, make_splits, data = rbind(SRE_train_PG, SRE_test_PG))
    
    SRE_split_AUC <- manual_rset(splits_AUC, c("Split SRE"))
    SRE_split_Brier <- manual_rset(splits_Brier, c("Split SRE"))
    SRE_split_PG <- manual_rset(splits_PG, c("Split SRE"))
    
    ####### 
    # Create recipes, for AUC, Brier and PG
    normalizable_AUC <- colnames(training(SRE_split_AUC$splits[[1]])[unlist(lapply(training(SRE_split_AUC$splits[[1]]), function(x) n_distinct(x)>2))])
    normalizable_Brier <- colnames(training(SRE_split_Brier$splits[[1]])[unlist(lapply(training(SRE_split_Brier$splits[[1]]), function(x) n_distinct(x)>2))])
    normalizable_PG <- colnames(training(SRE_split_PG$splits[[1]])[unlist(lapply(training(SRE_split_PG$splits[[1]]), function(x) n_distinct(x)>2))])
    
    SRE_recipe_AUC <- recipe(label~., data = training(SRE_split_AUC$splits[[1]])) %>%
      step_hai_winsorized_truncate(all_of(names(!!training(SRE_split_AUC$splits[[1]]))[!!winsorizable_AUC]), fraction = 0.025) %>%
      step_rm(all_of(names(!!training(SRE_split_AUC$splits[[1]]))[!!winsorizable_AUC])) %>%
      step_mutate_at(contains("winsorized"), fn = ~0.4 * ./ sd(.)) %>%
      step_mutate(across(where(is.logical), as.integer)) %>%
      step_normalize(all_of(setdiff(!!normalizable_AUC, colnames(!!training(SRE_split_AUC$splits[[1]])[!!winsorizable_AUC])))) %>%
      step_zv()
    
    SRE_recipe_Brier <- recipe(label~., data = training(SRE_split_Brier$splits[[1]])) %>%
      step_hai_winsorized_truncate(all_of(names(!!training(SRE_split_Brier$splits[[1]]))[!!winsorizable_Brier]), fraction = 0.025) %>%
      step_rm(all_of(names(!!training(SRE_split_Brier$splits[[1]]))[!!winsorizable_Brier])) %>%
      step_mutate_at(contains("winsorized"), fn = ~0.4 * ./ sd(.)) %>%
      step_mutate(across(where(is.logical), as.integer)) %>%
      step_normalize(all_of(setdiff(!!normalizable_Brier, colnames(!!training(SRE_split_Brier$splits[[1]])[!!winsorizable_Brier])))) %>%
      step_zv()
    
    SRE_recipe_PG <- recipe(label~., data = training(SRE_split_PG$splits[[1]])) %>%
      step_hai_winsorized_truncate(all_of(names(!!training(SRE_split_PG$splits[[1]]))[!!winsorizable_PG]), fraction = 0.025) %>%
      step_rm(all_of(names(!!training(SRE_split_PG$splits[[1]]))[!!winsorizable_PG])) %>%
      step_mutate_at(contains("winsorized"), fn = ~0.4 * ./ sd(.)) %>%
      step_mutate(across(where(is.logical), as.integer)) %>%
      step_normalize(all_of(setdiff(!!normalizable_PG, colnames(!!training(SRE_split_PG$splits[[1]])[!!winsorizable_PG])))) %>%
      step_zv()

    ####### 
    # Fit regular lasso for AUC, Brier, PG and extract metrics
    SRE_model_auc <- 
      parsnip::logistic_reg(
        mode = "classification",
        mixture = 1,
        penalty = lambda_1se_auc
      ) %>%
      set_engine("glmnet")
    
    SRE_wf_auc <- workflow() %>%
      #add_formula(label~.) %>%
      add_recipe(SRE_recipe_AUC) %>%
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
      add_recipe(SRE_recipe_Brier) %>%
      add_model(SRE_model_brier)    
    
    SRE_model_pg <- 
      parsnip::logistic_reg(
        mode = "classification",
        mixture = 1,
        penalty = best_lambda_pg    #sd can be very high for PG resulting in way too high lambda, leaving only the intercept
      ) %>%
      set_engine("glmnet")
    
    SRE_wf_pg <- workflow() %>%
      #add_formula(label~.) %>%
      add_recipe(SRE_recipe_PG) %>%
      add_model(SRE_model_pg)    

    final_SRE_fit_auc <- SRE_wf_auc %>% last_fit(SRE_split_AUC$splits[[1]], metrics = metrics)
    final_SRE_fit_brier <- SRE_wf_brier %>% last_fit(SRE_split_Brier$splits[[1]], metrics = metrics)
    final_SRE_fit_pg <- SRE_wf_pg %>% last_fit(SRE_split_PG$splits[[1]], metrics = metrics)
    
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
  write.csv(AUC_results, file = paste("./results/",dataset_vector[dataset_counter],"_AUC.csv", sep = ""))
  write.csv(Brier_results, file = paste("./results/",dataset_vector[dataset_counter],"_BRIER.csv", sep = ""))
  PG_results$metric<-unlist(PG_results$metric)
  write.csv(PG_results, file = paste("./results/",dataset_vector[dataset_counter],"_PG.csv", sep = ""))
  
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
