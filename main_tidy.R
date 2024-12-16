# main
if (!require("pacman")) install.packages("pacman") ; require("pacman")
p_load(glmnet, glmnetUtils, mgcv, MASS, tidyverse, xgboost, DiagrammeR, stringr, tictoc, doParallel, pROC, earth, Matrix, pre, caret, parsnip, ggplot2, recipes, rsample, workflows, healthyR.ai, rlang, yardstick, bonsai, lightgbm, ranger, tune, DescTools, rules, discrim, EMP, party, future)

#conventions:
# target variable = label
# indepentent variables = all others
# SRE bag -> rpart


# EMP toevoegen: EMP package
setwd("./github/SRE_credit_scoring")
source("./src/misc.R")
source("./src/pre.R")
source("./src/SRE.R")
source("./src/PLTR.R")
source("./src/hyperparameters.R")
source("./src/BigSummary.R")
source("./src/data_loader.R")
datasets <- load_data()

#parallel
cl <- makeCluster(detectCores())
registerDoParallel(cl)
plan(multisession, workers = availableCores())
options(future.globals.maxSize = 1000 * 10240^2)

#(AUCROC, Brier, partialGini)
metric = "AUCROC"
nr_repeats = 3
outerfolds = 2
fraction = 1 #subsampling fraction
dataset_vector = c("GC", "AC", "HMEQ", "TH02", "LC", "TC", "GMSC", "PAKDD", "BF")

metrics = metric_set(roc_auc, brier_class)

# create empty dataframe metric_results with columns: (dataset, repeat, fold, algorithm, metric)	
metric_results <- data.frame(
  dataset = character(),
  nr_fold = integer(),
  algorithm = character(),
  metric = double(),
  stringsAsFactors = FALSE
)
predictions <- list()

dataset_counter = 1

for(dataset in datasets[1]) {
  
  #subsampling
  set.seed(111)
  #dataset <- dataset[sample(nrow(dataset), round(fraction*nrow(dataset)), ), ]
  
  
  AUC_results <- metric_results
  Brier_results <- metric_results
  PG_results <- metric_results
  EMP_results <- metric_results
  
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
    
    set.seed(innerseed)
    inner_split <- train %>% validation_split(prop=3/4)
    inner_ids_in <- inner_split$splits[[1]]$in_id
    
    inner_train_bake <- XGB_recipe%>%prep(analysis(inner_split$splits[[1]]))%>%bake(analysis(inner_split$splits[[1]]))
    inner_test_bake <- XGB_recipe%>%prep(analysis(inner_split$splits[[1]]))%>%bake(assessment(inner_split$splits[[1]]))
    
    RE_recipe_inner <- recipe(label ~., data = inner_train_bake) %>%
      step_impute_mean(all_numeric_predictors()) %>%
      step_impute_mode(all_string_predictors()) %>%
      step_impute_mode(all_factor_predictors()) %>%
      step_novel(all_nominal_predictors()) %>%
      step_dummy(all_string_predictors()) %>%
      step_dummy(all_factor_predictors()) %>%
      step_zv(all_predictors())
    
    
    ctrl <- trainControl(method="LGOCV",
                         number=1, 
                         index = list(inner_ids_in), 
                         classProbs=TRUE, 
                         summaryFunction=BigSummary, 
                         search="grid", 
                         allowParallel=TRUE,
                         savePredictions = "all")
    
    
    ############################################################################
    # SINGLE CLASSIFIERS
    ############################################################################
    
    #####
    # LRR
    #####
    print("LRR")
    
    tic()
    LRR_model <- 
      parsnip::logistic_reg(
        mode = "classification",
        mixture = tune(),
        penalty = tune() #change
      ) %>%
      set_engine("glmnet")
    
    LRR_wf <- workflow() %>%
      add_recipe(LINEAR_recipe) %>%
      add_model(LRR_model)


    LRR_tuned <- tune::tune_grid(
      object = LRR_wf,
      resamples = inner_split,
      grid = hyperparameters_LR_R_tidy, 
      metrics = metrics,
      control = tune::control_grid(verbose = TRUE, save_pred = TRUE, parallel_over = "everything")
    )
    toc()
    
    best_model_auc <- LRR_tuned %>% select_best(metric="roc_auc")
    final_LRR_wf_auc <- LRR_wf %>% finalize_workflow(best_model_auc)
    final_LRR_fit_auc <- final_LRR_wf_auc %>% last_fit(folds$splits[[i]], metrics = metrics)

    best_model_brier <- LRR_tuned %>% select_best(metric="brier_class")
    final_LRR_wf_brier <- LRR_wf %>% finalize_workflow(best_model_brier)
    final_LRR_fit_brier <- final_LRR_wf_brier %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    best_model_pg <- LRR_tuned %>% select_best_pg_LRR()
    final_LRR_wf_pg <- LRR_wf %>% finalize_workflow(best_model_pg)
    final_LRR_fit_pg <- final_LRR_wf_pg %>% last_fit(folds$splits[[i]], metrics = metrics)

    best_model_emp <- LRR_tuned %>% select_best_emp_LRR()
    final_LRR_wf_emp <- LRR_wf %>% finalize_workflow(best_model_emp)
    final_LRR_fit_emp <- final_LRR_wf_emp %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    #Save predictions
    LRR_predictions_AUC <- final_LRR_fit_auc$.predictions[[1]]$.pred_X1
    LRR_predictions_Brier <- final_LRR_fit_brier$.predictions[[1]]$.pred_X1
    LRR_predictions_PG <- final_LRR_fit_pg$.predictions[[1]]$.pred_X1
    LRR_predictions_EMP <- final_LRR_fit_emp$.predictions[[1]]$.pred_X1
    
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
    
    emp <- final_LRR_fit_pg %>%
      collect_emp()
    EMP_results[nrow(EMP_results) + 1,] = list(dataset_vector[dataset_counter], i, "LRR", emp)
    
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
                                   ifelse(names(train_processed%>%dplyr::select(-label)) %in% smooth_vars, ", bs=\"cr\" )",""),
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
    
    GAM_predictions <- final_GAM_fit$.predictions[[1]]$.pred_X1

    
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
    
    emp <- final_GAM_fit %>%
      collect_emp()
    EMP_results[nrow(EMP_results) + 1,] = list(dataset_vector[dataset_counter], i, "GAM", emp)
    
    
    
    
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
    
    LDA_predictions <- LDA_preds$X1
    
    #AUC
    g <- roc(label ~ X1, data = LDA_preds, direction = "<")
    AUC <- g$auc
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "LDA", AUC)

    brier <- brier_score(preds = LDA_preds$X1, truth = LDA_preds$label)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "LDA", brier)
    
    pg <- partialGini(actuals = LDA_preds$label, preds = LDA_preds$X1)    
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "LDA", pg)
    
    emp <- empCreditScoring(classes = LDA_preds$label, scores = LDA_preds$X1)$EMPC    
    EMP_results[nrow(EMP_results) + 1,] = list(dataset_vector[dataset_counter], i, "LDA", emp)
    
    
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
    CTREE_predictions_AUC <- CTREE_preds$X1
    
    
    g <- roc(label ~ X1, data = CTREE_preds, direction = "<")
    AUC <- g$auc
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "CTREE", AUC)
    
    #Brier
    
    CTREE_model_Brier <- train(TREE_recipe, data = train, method = "ctree",
                               tuneGrid = expand.grid(mincriterion = (CTREE_model$results%>%slice_min(Brier)%>%dplyr::select(mincriterion))[[1]])) 
    CTREE_preds <- predict(CTREE_model_Brier, test, type = 'prob')
    CTREE_preds$label <- test$label
    CTREE_predictions_Brier <- CTREE_preds$X1
    
    
    brier <- brier_score(truth = CTREE_preds$label, preds = CTREE_preds$X1)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "CTREE", brier)
    
    #PG
    
    CTREE_model_PG <- train(TREE_recipe, data = train, method = "ctree",
                            tuneGrid = expand.grid(mincriterion = (CTREE_model$results%>%slice_max(partialGini)%>%dplyr::select(mincriterion))[[1]]))
    CTREE_preds <- predict(CTREE_model_PG, test, type = 'prob')
    CTREE_preds$label <- test$label
    CTREE_predictions_PG <- CTREE_preds$X1
    
    pg <- partialGini(CTREE_preds$X1, CTREE_preds$label)
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "CTREE", pg)
    
    #EMP
    mincriterion_EMP <- select_best_emp_CTREE(CTREE_model)
    
    CTREE_model_EMP <- train(TREE_recipe, data = train, method = "ctree",
                            tuneGrid = expand.grid(mincriterion = (mincriterion_EMP[[1]])))
    CTREE_preds <- predict(CTREE_model_EMP, test, type = 'prob')
    CTREE_preds$label <- test$label
    CTREE_predictions_EMP <- CTREE_preds$X1
    
    EMP <- empCreditScoring(CTREE_preds$X1, CTREE_preds$label)$EMPC
    EMP_results[nrow(EMP_results) + 1,] = list(dataset_vector[dataset_counter], i, "CTREE", EMP)
    
    
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
      resamples = inner_split,
      grid = hyperparameters_RF_tidy,
      metrics = metrics,
      control = tune::control_grid(verbose = TRUE, save_pred = TRUE, parallel_over = "everything")
    )
    
    
    best_model_auc <- RF_tuned %>% select_best(metric="roc_auc")
    final_RF_wf_auc <- RF_wf %>% finalize_workflow(best_model_auc)
    final_RF_fit_auc <- final_RF_wf_auc %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    best_model_brier <- RF_tuned %>% select_best(metric="brier_class")
    final_RF_wf_brier <- RF_wf %>% finalize_workflow(best_model_brier)
    final_RF_fit_brier <- final_RF_wf_brier %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    best_model_pg <- RF_tuned %>% select_best_pg_RF()
    final_RF_wf_pg <- RF_wf %>% finalize_workflow(best_model_pg)
    final_RF_fit_pg <- final_RF_wf_pg %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    best_model_emp <- RF_tuned %>% select_best_emp_RF()
    final_RF_wf_emp <- RF_wf %>% finalize_workflow(best_model_emp)
    final_RF_fit_emp <- final_RF_wf_emp %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    #Save predictions
    RF_predictions_AUC <- final_RF_fit_auc$.predictions[[1]]$.pred_X1
    RF_predictions_Brier <- final_RF_fit_brier$.predictions[[1]]$.pred_X1
    RF_predictions_PG <- final_RF_fit_pg$.predictions[[1]]$.pred_X1
    RF_predictions_EMP <- final_RF_fit_emp$.predictions[[1]]$.pred_X1
    
    
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
    
    emp <- final_RF_fit_emp %>%
      collect_emp()
    EMP_results[nrow(EMP_results) + 1,] = list(dataset_vector[dataset_counter], i, "RF", emp)
    
    
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
        loss_reduction = tune()
      ) %>%
      set_engine("xgboost")
    
    xgb_wf <- workflow() %>%
      add_recipe(XGB_recipe) %>%
      add_model(xgb_model)

    xgb_tuned <- tune::tune_grid(
      object = xgb_wf,
      resamples = inner_split,
      grid = hyperparameters_XGB_tidy,
      metrics = metrics,
      control = tune::control_grid(verbose = TRUE, save_pred = TRUE, parallel_over = "everything")
    )

    best_booster_auc <- xgb_tuned %>% select_best(metric="roc_auc")
    final_xgb_wf_auc <- xgb_wf %>% finalize_workflow(best_booster_auc)
    final_xgb_fit_auc <- final_xgb_wf_auc %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    best_booster_brier <- xgb_tuned %>% select_best(metric="brier_class")
    final_xgb_wf_brier <- xgb_wf %>% finalize_workflow(best_booster_brier)
    final_xgb_fit_brier <- final_xgb_wf_brier %>% last_fit(folds$splits[[i]], metrics = metrics)

    best_booster_pg <- xgb_tuned %>% select_best_pg_XGB()
    final_xgb_wf_pg <- xgb_wf %>% finalize_workflow(best_booster_pg)
    final_xgb_fit_pg <- final_xgb_wf_pg %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    best_booster_emp <- xgb_tuned %>% select_best_emp_XGB()
    final_xgb_wf_emp <- xgb_wf %>% finalize_workflow(best_booster_emp)
    final_xgb_fit_emp <- final_xgb_wf_emp %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    
    #Save predictions
    XGB_predictions_AUC <- final_xgb_fit_auc$.predictions[[1]]$.pred_X1
    XGB_predictions_Brier <- final_xgb_fit_brier$.predictions[[1]]$.pred_X1
    XGB_predictions_PG <- final_xgb_fit_pg$.predictions[[1]]$.pred_X1
    XGB_predictions_EMP <- final_xgb_fit_emp$.predictions[[1]]$.pred_X1
    
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

    emp <- final_xgb_fit_emp %>%
      collect_emp()
    EMP_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "XGB", emp)
    
    
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
        sample_size = tune(),
        loss_reduction = tune()
      ) %>%
      set_engine("lightgbm") %>%
      translate()
    
    lgbm_wf <- workflow() %>%
      add_recipe(XGB_recipe) %>%
      add_model(lgbm_model)
    
    lgbm_tuned <- tune::tune_grid(
      object = lgbm_wf,
      resamples = inner_split, 
      grid = hyperparameters_LGBM_tidy, 
      metrics = metrics,
      control = tune::control_grid(verbose = TRUE, save_pred = TRUE, parallel_over = "everything")
    )
    
    best_booster_auc <- lgbm_tuned %>% select_best(metric="roc_auc")
    final_lgbm_wf_auc <- lgbm_wf %>% finalize_workflow(best_booster_auc)
    final_lgbm_fit_auc <- final_lgbm_wf_auc %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    best_booster_brier <- lgbm_tuned %>% select_best(metric="brier_class")
    final_lgbm_wf_brier <- lgbm_wf %>% finalize_workflow(best_booster_brier)
    final_lgbm_fit_brier <- final_lgbm_wf_brier %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    best_model_pg <- lgbm_tuned %>% select_best_pg_LGBM()
    final_lgbm_wf_pg <- lgbm_wf %>% finalize_workflow(best_model_pg)
    final_lgbm_fit_pg <- final_lgbm_wf_pg %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    best_model_emp <- lgbm_tuned %>% select_best_emp_LGBM()
    final_lgbm_wf_emp <- lgbm_wf %>% finalize_workflow(best_model_emp)
    final_lgbm_fit_emp <- final_lgbm_wf_emp %>% last_fit(folds$splits[[i]], metrics = metrics)
    
    
    #Save predictions
    LGBM_predictions_AUC <- final_lgbm_fit_auc$.predictions[[1]]$.pred_X1
    LGBM_predictions_Brier <- final_lgbm_fit_brier$.predictions[[1]]$.pred_X1
    LGBM_predictions_PG <- final_lgbm_fit_pg$.predictions[[1]]$.pred_X1
    LGBM_predictions_EMP <- final_lgbm_fit_emp$.predictions[[1]]$.pred_X1
    
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

    emp <- final_lgbm_fit_emp %>%
      collect_emp()
    EMP_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "LGBM", emp)
    
    ############################################################################
    # HETEROGENEOUS (RULE) ENSEMBLES 
    ############################################################################
    
    #####
    # RE boosting
    #####
    print("RE: boosting")

    set.seed(innerseed)
    tic()
    
    #try lambda min instead of 1se
    RE_model_boosting <- train(XGB_recipe, data = train, method = "pre",
                      ntrees = 100, tree.unbiased = FALSE, family = "binomial", trControl = ctrl,
                      tuneGrid = preGrid_boosting, singleconditions = FALSE,
                      winsfrac = 0.05, normalize = TRUE, #same a priori influence as a typical rule
                      ad.alpha = 0,
                      verbose = TRUE,
                      metric = "AUCROC", allowParallel = FALSE,
                      par.init=FALSE,
                      par.final=FALSE,
                      nfolds=3)    
    toc()
    #AUC
    RE_preds_boosting <- predict(RE_model_boosting, test, type = 'probs')
    RE_preds_boosting$label <- test$label
    
    #Save predictions
    RE_boosting_predictions_AUC <- RE_preds_boosting$X1
    
    g <- roc(label ~ X1, data = RE_preds_boosting, direction = "<")
    AUC <- g$auc
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "RE_boosting", AUC)
    
    #Brier
    RE_model_boosting_Brier <- train(XGB_recipe, data = train, method = "pre",
                            ntrees = 100, family = "binomial", trControl = trainControl(method = "none", classProbs = TRUE),
                            tuneGrid = getModelInfo("pre")[[1]]$grid( 
                              maxdepth = (RE_model_boosting$results%>%slice_min(Brier)%>%dplyr::select(maxdepth))[[1]][1],
                              learnrate = (RE_model_boosting$results%>%slice_min(Brier)%>%dplyr::select(learnrate))[[1]][1],
                              penalty.par.val = c("lambda.min"), # λand γ combination yielding the sparsest solution within 1 standard error of the error criterion of the minimum is returned
                              sampfrac = 1,
                              use.grad = TRUE), ad.alpha = 0, tree.unbiased = FALSE, singleconditions = FALSE,
                            winsfrac = 0.05, normalize = TRUE, #same a priori influence as a typical rule
                            verbose = TRUE,
                            allowParallel = TRUE,
                            par.init=TRUE,
                            par.final=TRUE,
                            nfolds = 3)
    RE_preds_boosting <- predict(RE_model_boosting_Brier, test, type = 'prob')
    RE_preds_boosting$label <- test$label
    
    #Save predictions
    RE_boosting_predictions_Brier <- RE_preds_boosting$X1
    
    brier <- brier_score(truth = RE_preds_boosting$label, preds = RE_preds_boosting$X1)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "RE_boosting", brier)
    
    #PG
    RE_model_boosting_PG <- train(XGB_recipe, data = train, method = "pre",
                            ntrees = min(100, round(nrow(train)/2)), family = "binomial", trControl = trainControl(method = "none", classProbs = TRUE),
                            tuneGrid = getModelInfo("pre")[[1]]$grid( 
                              maxdepth = (RE_model_boosting$results%>%slice_max(partialGini)%>%dplyr::select(maxdepth))[[1]][1],
                              learnrate = (RE_model_boosting$results%>%slice_max(partialGini)%>%dplyr::select(learnrate))[[1]][1],
                              penalty.par.val = c("lambda.min"), # λand γ combination yielding the sparsest solution within 1 standard error of the error criterion of the minimum is returned
                              sampfrac = 1,
                              use.grad = TRUE), tree.unbiased = FALSE, ad.alpha = 0, singleconditions = FALSE,
                            winsfrac = 0.05, normalize = TRUE, #same a priori influence as a typical rule
                            verbose = TRUE,
                            metric = "AUCROC", allowParallel = TRUE,
                            par.init=TRUE,
                            par.final=TRUE,
                            nfolds = 3)
    RE_preds_boosting <- predict(RE_model_boosting_PG, test, type = 'prob')
    RE_preds_boosting$label <- test$label
    
    #Save predictions
    RE_boosting_predictions_PG <- RE_preds_boosting$X1
    
    pg <- partialGini(RE_preds_boosting$X1, RE_preds_boosting$label)
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "RE_boosting", pg)
    
    #EMP
    RE_boosting_EMP_params <- select_best_emp_RE_boosting(RE_model_boosting)
    RE_model_boosting_EMP <- train(XGB_recipe, data = train, method = "pre",
                                  ntrees = min(100, round(nrow(train)/2)), family = "binomial", trControl = trainControl(method = "none", classProbs = TRUE),
                                  tuneGrid = getModelInfo("pre")[[1]]$grid( 
                                    maxdepth = (RE_boosting_EMP_params%>%dplyr::select(maxdepth))[[1]][1],
                                    learnrate = (RE_boosting_EMP_params%>%dplyr::select(learnrate))[[1]][1],
                                    penalty.par.val = c("lambda.min"), # λand γ combination yielding the sparsest solution within 1 standard error of the error criterion of the minimum is returned
                                    sampfrac = 1,
                                    use.grad = TRUE), tree.unbiased = FALSE, ad.alpha = 0, singleconditions = FALSE,
                                  winsfrac = 0.05, normalize = TRUE, #same a priori influence as a typical rule
                                  verbose = TRUE,
                                  metric = "AUCROC", allowParallel = TRUE,
                                  par.init=TRUE,
                                  par.final=TRUE)
    RE_preds_boosting <- predict(RE_model_boosting_EMP, test, type = 'prob')
    RE_preds_boosting$label <- test$label
    
    #Save predictions
    RE_boosting_predictions_EMP <- RE_preds_boosting$X1
     
    emp <- empCreditScoring(RE_preds_boosting$X1, RE_preds_boosting$label)$EMPC
    EMP_results[nrow(EMP_results) + 1,] = list(dataset_vector[dataset_counter], i, "RE_boosting", emp)
    
    
    
    #####
    # RE RF
    #####
    
    # same as above but with RF instead of boosting
    # Later: change to biased tree to employ rpart when available in package
    print("RE: RF")

    
    ######
    # BETTER?
    pre_rf_results <- list()
    pre_rf_resulting_models <- list()
    tic()
    for(config in 1:nrow(preGrid_RF)) {
      cat("RE config: ", config, "/", nrow(preGrid_RF))
      inner_RE_model_RF <- pre_custom(formula = label~.,
        data = RE_recipe_inner%>%prep(inner_train_bake)%>%bake(inner_train_bake),
        family = "binomial",
        ad.alpha = 0, #ridge estimates for adalasso
        winsfrac = 0.05,
        normalize = TRUE,
        removecomplements = TRUE,
        removeduplicates = TRUE,
        par.init = TRUE,
        par.final = TRUE,
        verbose = TRUE,
        tree.unbiased = FALSE,
        ntrees = 100,
        sampfrac = preGrid_RF$sampfrac[config],
        maxdepth = preGrid_RF$maxdepth[config],
        learnrate = preGrid_RF$learnrate[config],
        mtry = preGrid_RF$mtry[config],
        use.grad = preGrid_RF$use.grad[config]
        #ad.penalty = preGrid_RF$penalty.par.val[config]
    )
      pre_rf_resulting_models[[config]] <- inner_RE_model_RF
      pre_rf_results[[config]] <- predict(inner_RE_model_RF, RE_recipe_inner%>%prep(inner_train_bake)%>%bake(inner_test_bake), type = 'response')
    }
    toc()
    
    inner_re_rf_AUC <- inner_re_rf_Brier <- inner_re_rf_PG <- inner_re_rf_EMP <- list()
    for(results_count in 1:length(pre_rf_results)) {
      inner_re_rf_AUC[results_count] <- roc(inner_test_bake$label, pre_rf_results[[results_count]])$auc
      inner_re_rf_Brier[results_count] <- brier_score(truth = inner_test_bake$label, preds = pre_rf_results[[results_count]])
      inner_re_rf_PG[results_count] <- partialGini(pre_rf_results[[results_count]], inner_test_bake$label)
      inner_re_rf_EMP[results_count] <- empCreditScoring(pre_rf_results[[results_count]], inner_test_bake$label)$EMPC
    }
    
    # Identify indices with max values for each metric
    max_indices <- list(
      AUC = which.max(inner_re_rf_AUC),
      Brier = which.min(inner_re_rf_Brier),
      PG = which.max(inner_re_rf_PG),
      EMP = which.max(inner_re_rf_EMP)
    )
    
    # ONLY USEFUL WHEN UTILIZING SINGLE HOLDOUT SET
    
    RE_RF_inner_model_AUC <- pre_rf_resulting_models[[which.max(inner_re_rf_AUC)]]
    RE_RF_inner_model_Brier <- pre_rf_resulting_models[[which.max(inner_re_rf_Brier)]]
    RE_RF_inner_model_PG <- pre_rf_resulting_models[[which.max(inner_re_rf_PG)]]
    RE_RF_inner_model_EMP <- pre_rf_resulting_models[[which.max(inner_re_rf_EMP)]]
    RE_RF_inner_models_list <- list(RE_RF_inner_model_AUC, RE_RF_inner_model_Brier, RE_RF_inner_model_PG, RE_RF_inner_model_EMP) 
    
    # Identify unique configurations to avoid redundant training
    unique_configs <- unique(unlist(max_indices))
    config_map <- data.frame(
      metric = names(max_indices),
      index = unlist(max_indices)
    )
    
    # Initialize a list to store models for unique configurations
    trained_models <- list()
    
    # Train models for each unique configuration
    for (idx in unique_configs) {
      trained_models[[as.character(idx)]] <- pre_custom(
        formula = label~.,
        data = XGB_recipe %>% prep(train) %>% bake(train),
        family = "binomial",
        ad.alpha = 0,
        winsfrac = 0.05,
        normalize = TRUE,
        removecomplements = TRUE,
        removeduplicates = TRUE,
        par.init = TRUE,
        par.final = TRUE,
        verbose = TRUE,
        tree.unbiased = FALSE,
        ntrees = 100,
        sampfrac = preGrid_RF$sampfrac[idx],
        maxdepth = preGrid_RF$maxdepth[idx],
        learnrate = preGrid_RF$learnrate[idx],
        mtry = preGrid_RF$mtry[idx],
        use.grad = preGrid_RF$use.grad[idx]
      )
    }
    
    # Assign trained models to variables based on the config map
    RE_model_RF_AUC <- trained_models[[as.character(max_indices$AUC)]]
    RE_model_RF_Brier <- trained_models[[as.character(max_indices$Brier)]]
    RE_model_RF_PG <- trained_models[[as.character(max_indices$PG)]]
    RE_model_RF_EMP <- trained_models[[as.character(max_indices$EMP)]]

    
    ######
    
    #AUC
    RE_preds_RF <- data.frame("X1"=predict(RE_model_RF_AUC, test_bake, type = 'response'))
    RE_preds_RF$label <- test$label
    
    #Save predictions
    RE_RF_predictions_AUC <- RE_preds_RF$X1
    
    g <- roc(label ~ X1, data = RE_preds_RF, direction = "<")
    AUC <- g$auc
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "RE_RF", AUC)
    
    #Brier
    RE_preds_RF <- data.frame("X1"=predict(RE_model_RF_Brier, test_bake, type = 'response'))
    RE_preds_RF$label <- test$label
    
    #Save predictions
    RE_RF_predictions_Brier <- RE_preds_RF$X1
    
    brier <- brier_score(truth = RE_preds_RF$label, preds = RE_preds_RF$X1)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "RE_RF", brier)
    
    #PG
    RE_preds_RF <- data.frame("X1"=predict(RE_model_RF_PG, test_bake, type = 'response'))
    RE_preds_RF$label <- test$label
    
    #Save predictions
    RE_RF_predictions_PG <- RE_preds_RF$X1
    
    pg <- partialGini(RE_preds_RF$X1, RE_preds_RF$label)
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "RE_RF", pg)
    
    #EMP
    RE_preds_RF <- data.frame("X1"=predict(RE_model_RF_EMP, test_bake, type = 'response'))
    RE_preds_RF$label <- test$label
    
    #Save predictions
    RE_RF_predictions_EMP <- RE_preds_RF$X1
    
    emp <- empCreditScoring(RE_preds_RF$X1, RE_preds_RF$label)$EMPC
    EMP_results[nrow(EMP_results) + 1,] = list(dataset_vector[dataset_counter], i, "RE_RF", emp)
    
    
    #####
    # RE bag
    #####
    
    # same as above but with bagging
    print("RE: bag")
    
    set.seed(innerseed)
    RE_model_bag <- train(XGB_recipe, data = train, method = "pre",
                         ntrees = 100, family = "binomial", trControl = trainControl(method = "none", classProbs = TRUE),
                         tuneGrid = preGrid_bag, ad.alpha = 0, tree.unbiased = FALSE, 
                         singleconditions = FALSE,
                         winsfrac = 0.05, normalize = TRUE, #same a priori influence as a typical rule
                         verbose = TRUE,
                         metric = "AUCROC", allowParallel = TRUE,
                         par.init=TRUE,
                         par.final=TRUE
                         )    
    
    #AUC
    RE_preds_bag <- predict(RE_model_bag, test, type = 'prob')
    RE_preds_bag$label <- test$label
    
    #Save predictions
    RE_bag_predictions <- RE_preds_bag$X1
    
    g <- roc(label ~ X1, data = RE_preds_bag, direction = "<")
    AUC <- g$auc
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "RE_bag", AUC)
    
    #Brier
    brier <- brier_score(truth = RE_preds_bag$label, preds = RE_preds_bag$X1)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "RE_bag", brier)
    
    #PG
    pg <- partialGini(RE_preds_bag$X1, RE_preds_bag$label)
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "RE_bag", pg)
    
    #EMP
    emp <- empCreditScoring(RE_preds_bag$X1, RE_preds_bag$label)$EMPC
    EMP_results[nrow(EMP_results) + 1,] = list(dataset_vector[dataset_counter], i, "RE_bag", emp)
    
    ######
    # SRE
    ######
    source("./src/SRE.R")
    print("SRE: random forest")
    
    SRE_RF <- cv.SRE(inner_split,
                     tree_algorithm = "randomforest",
                     RE_model_AUC = RE_model_RF_AUC,
                     RE_model_Brier = RE_model_RF_Brier,
                     RE_model_PG = RE_model_RF_PG,
                     RE_model_EMP = RE_model_RF_EMP,
                     inner_RF_list = RE_RF_inner_models_list,
                     GAM_recipe = GAM_recipe,
                     metrics = metrics,
                     train_bake = train_bake,
                     test_bake = test_bake,
                     regularization = NULL
                     )
    
    #Save predictions
    SRE_RF_predictions_AUC <- SRE_RF$best_AUC$.predictions[[1]]$.pred_X1
    SRE_RF_predictions_Brier <- SRE_RF$best_Brier$.predictions[[1]]$.pred_X1
    SRE_RF_predictions_PG <- SRE_RF$best_PG$.predictions[[1]]$.pred_X1
    SRE_RF_predictions_EMP <- SRE_RF$best_EMP$.predictions[[1]]$.pred_X1
    
    ######
    # Extract metrics
    auc <- SRE_RF$best_AUC %>%
      collect_metrics() %>%
      filter(.metric == "roc_auc") %>%
      pull(.estimate)
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE_RF", auc)
    
    brier <- SRE_RF$best_Brier %>%
      collect_metrics() %>%
      filter(.metric == "brier_class") %>%
      pull(.estimate)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE_RF", brier)
    
    pg <- SRE_RF$best_PG %>%
      collect_pg()
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE_RF", pg)
    
    emp <- SRE_RF$best_EMP %>%
      collect_emp()
    EMP_results[nrow(EMP_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE_RF", emp)
    
    print("SRE: bagging")
    
    SRE_bag <- cv.SRE(inner_split,
                     tree_algorithm = "bagging",
                     RE_model_AUC = RE_model_bag,
                     RE_model_Brier = RE_model_bag,
                     RE_model_PG = RE_model_bag,
                     RE_model_EMP = RE_model_bag,
                     GAM_recipe = GAM_recipe,
                     metrics = metrics,
                     train_bake = train_bake,
                     test_bake = test_bake,
                     regularization = NULL
    )
    
    #Save predictions
    SRE_bag_predictions_AUC <- SRE_bag$best_AUC$.predictions[[1]]$.pred_X1
    SRE_bag_predictions_Brier <- SRE_bag$best_Brier$.predictions[[1]]$.pred_X1
    SRE_bag_predictions_PG <- SRE_bag$best_PG$.predictions[[1]]$.pred_X1
    SRE_bag_predictions_EMP <- SRE_bag$best_EMP$.predictions[[1]]$.pred_X1
    
    ######
    # Extract metrics
    auc <- SRE_bag$best_AUC %>%
      collect_metrics() %>%
      filter(.metric == "roc_auc") %>%
      pull(.estimate)
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE_bag", auc)
    
    brier <- SRE_bag$best_Brier %>%
      collect_metrics() %>%
      filter(.metric == "brier_class") %>%
      pull(.estimate)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE_bag", brier)
    
    pg <- SRE_bag$best_PG %>%
      collect_pg()
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE_bag", pg)
    
    emp <- SRE_bag$best_EMP %>%
      collect_emp()
    EMP_results[nrow(EMP_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE_bag", emp)
    
    
    print("SRE: boosting")
    
    SRE_boosting <- cv.SRE(inner_split,
                     tree_algorithm = "boosting",
                     RE_model_AUC = RE_model_boosting,
                     RE_model_Brier = RE_model_boosting_Brier,
                     RE_model_PG = RE_model_boosting_PG,
                     RE_model_EMP = RE_model_boosting_EMP,
                     GAM_recipe = GAM_recipe,
                     metrics = metrics,
                     train_bake = train_bake,
                     test_bake = test_bake,
                     regularization = NULL
    )
    
    #Save predictions
    SRE_boosting_predictions_AUC <- SRE_boosting$best_AUC$.predictions[[1]]$.pred_X1
    SRE_boosting_predictions_Brier <- SRE_boosting$best_Brier$.predictions[[1]]$.pred_X1
    SRE_boosting_predictions_PG <- SRE_boosting$best_PG$.predictions[[1]]$.pred_X1
    SRE_boosting_predictions_EMP <- SRE_boosting$best_EMP$.predictions[[1]]$.pred_X1
    
    ######
    # Extract metrics
    auc <- SRE_boosting$best_AUC %>%
      collect_metrics() %>%
      filter(.metric == "roc_auc") %>%
      pull(.estimate)
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE_boosting", auc)
    
    brier <- SRE_boosting$best_Brier %>%
      collect_metrics() %>%
      filter(.metric == "brier_class") %>%
      pull(.estimate)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE_boosting", brier)
    
    pg <- SRE_boosting$best_PG %>%
      collect_pg()
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE_boosting", pg)
    
    emp <- SRE_boosting$best_EMP %>%
      collect_emp()
    EMP_results[nrow(EMP_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE_boosting", emp)
    
    print("SRE: PLTR")
    
    SRE_PLTR <- cv.SRE(inner_split,
                           tree_algorithm = "PLTR",
                           RE_model_AUC = NULL,
                           RE_model_Brier = NULL,
                           RE_model_PG = NULL,
                           RE_model_EMP = NULL,
                           GAM_recipe = GAM_recipe,
                           metrics = metrics,
                           train_bake = train_bake,
                           test_bake = test_bake,
                           regularization = NULL
    )
    
    #Save predictions
    SRE_PLTR_predictions_AUC <- SRE_PLTR$best_AUC$.predictions[[1]]$.pred_X1
    SRE_PLTR_predictions_Brier <- SRE_PLTR$best_Brier$.predictions[[1]]$.pred_X1
    SRE_PLTR_predictions_PG <- SRE_PLTR$best_PG$.predictions[[1]]$.pred_X1
    SRE_PLTR_predictions_EMP <- SRE_PLTR$best_EMP$.predictions[[1]]$.pred_X1
    
    ######
    # Extract metrics
    auc <- SRE_PLTR$best_AUC %>%
      collect_metrics() %>%
      filter(.metric == "roc_auc") %>%
      pull(.estimate)
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE_PLTR", auc)
    
    brier <- SRE_PLTR$best_Brier %>%
      collect_metrics() %>%
      filter(.metric == "brier_class") %>%
      pull(.estimate)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE_PLTR", brier)
    
    pg <- SRE_PLTR$best_PG %>%
      collect_pg()
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE_PLTR", pg)
    
    emp <- SRE_PLTR$best_EMP %>%
      collect_emp()
    EMP_results[nrow(EMP_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE_PLTR", emp)
    
    ######
    # PLTR
    ######
    
    source("./src/PLTR.R")
    
    print("PLTR")
    PLTR <- cv.PLTR(inner_split, metrics, train_bake, test_bake)
    
    auc <- PLTR$best_AUC %>%
      collect_metrics() %>%
      filter(.metric == "roc_auc") %>%
      pull(.estimate)
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "PLTR", auc)
    
    brier <- PLTR$best_Brier %>%
      collect_metrics() %>%
      filter(.metric == "brier_class") %>%
      pull(.estimate)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "PLTR", brier)
    
    pg <- PLTR$best_PG %>%
      collect_pg()
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "PLTR", pg)
    
    emp <- PLTR$best_EMP %>%
      collect_emp()
    EMP_results[nrow(EMP_results) + 1,] = list(dataset_vector[dataset_counter], i, "PLTR", emp)
    
    # Predictions
    PLTR_predictions_AUC <- PLTR$best_AUC$.predictions[[1]]$.pred_X1
    PLTR_predictions_Brier <- PLTR$best_Brier$.predictions[[1]]$.pred_X1
    PLTR_predictions_PG <- PLTR$best_PG$.predictions[[1]]$.pred_X1
    PLTR_predictions_EMP <- PLTR$best_EMP$.predictions[[1]]$.pred_X1
    
    
    predictions_AUC = cbind(LRR_predictions_AUC, GAM_predictions, LDA_predictions, CTREE_predictions_AUC, RF_predictions_AUC, XGB_predictions_AUC, LGBM_predictions_AUC, RE_RF_predictions_AUC, RE_boosting_predictions_AUC, RE_bag_predictions, PLTR_predictions_AUC, SRE_RF_predictions_AUC, SRE_bag_predictions_AUC, SRE_boosting_predictions_AUC, SRE_PLTR_predictions_AUC)
    predictions_Brier = cbind(LRR_predictions_Brier, GAM_predictions, LDA_predictions, CTREE_predictions_Brier, RF_predictions_Brier, XGB_predictions_Brier, LGBM_predictions_Brier, RE_RF_predictions_Brier, RE_boosting_predictions_Brier, RE_bag_predictions, PLTR_predictions_Brier, SRE_RF_predictions_Brier, SRE_bag_predictions_Brier, SRE_boosting_predictions_AUC, SRE_PLTR_predictions_Brier)
    predictions_PG = cbind(LRR_predictions_PG, GAM_predictions, LDA_predictions, CTREE_predictions_PG, RF_predictions_PG, XGB_predictions_PG, LGBM_predictions_PG, RE_RF_predictions_PG, RE_boosting_predictions_PG, RE_bag_predictions, PLTR_predictions_PG, SRE_RF_predictions_PG, SRE_bag_predictions_PG, SRE_boosting_predictions_AUC, SRE_PLTR_predictions_PG)
    predictions_EMP = cbind(LRR_predictions_EMP, GAM_predictions, LDA_predictions, CTREE_predictions_EMP, RF_predictions_EMP, XGB_predictions_EMP, LGBM_predictions_EMP, RE_RF_predictions_EMP, RE_boosting_predictions_EMP, RE_bag_predictions, PLTR_predictions_EMP, SRE_RF_predictions_EMP, SRE_bag_predictions_EMP, SRE_boosting_predictions_AUC, SRE_PLTR_predictions_EMP)
    write.csv(predictions_AUC, file = paste("./predictions/",dataset_vector[dataset_counter],"_predictions_repeat_", i, "_AUC.csv", sep = ""))
    write.csv(predictions_Brier, file = paste("./predictions/",dataset_vector[dataset_counter],"_predictions_repeat_", i, "_Brier.csv", sep = ""))
    write.csv(predictions_PG, file = paste("./predictions/",dataset_vector[dataset_counter],"_predictions_repeat_", i, "_PG.csv", sep = ""))
    write.csv(predictions_EMP, file = paste("./predictions/",dataset_vector[dataset_counter],"_predictions_repeat_", i, "_EMP.csv", sep = ""))

  }
  
  write.csv(AUC_results, file = paste("./results/",dataset_vector[dataset_counter],"_v2_AUC_boost_rerun.csv", sep = ""))
  write.csv(Brier_results, file = paste("./results/",dataset_vector[dataset_counter],"_v2_BRIER_boost_rerun.csv", sep = ""))
  PG_results$metric<-unlist(PG_results$metric)
  write.csv(PG_results, file = paste("./results/",dataset_vector[dataset_counter],"_v2_PG_boost_rerun.csv", sep = ""))
  EMP_results <- drop_na(EMP_results)
  EMP_results$metric<-unlist(EMP_results$metric)
  write.csv(EMP_results, file = paste("./results/",dataset_vector[dataset_counter],"_v2_EMP_boost_rerun.csv", sep = ""))
  
  dataset_counter <- dataset_counter + 1
}

stopCluster(cl)
