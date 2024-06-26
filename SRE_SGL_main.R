# main
if (!require("pacman")) install.packages("pacman") ; require("pacman")
p_load(glmnet, glmnetUtils, mgcv, MASS, tidyverse, xgboost, DiagrammeR, stringr, tictoc, doParallel, pROC, earth, Matrix, pre, caret, parsnip, ggplot2, recipes, rsample, workflows, healthyR.ai, rlang, yardstick, bonsai, lightgbm, ranger, tune, DescTools, rules, discrim, sparsegl)
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
    
    set.seed(i)
    inner_folds <- train %>% vfold_cv(v=5)
    
    
    
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
    
   
    ############################################################################
    # HETEROGENEOUS (RULE) ENSEMBLES 
    ############################################################################
    source("./src/hyperparameters.R")
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
    
    
    ####### 
    #SRE_SGL
    #######
    # inner SRE loop for hyperparameter tuning (glmnet)
    
    lambdas_AUC <- c()
    lambdas_Brier <- c()
    lambdas_PG <- c()
    
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
      
      final_GAM_fit_inner <- GAM_wf %>% last_fit(inner_folds$splits[[i]], metrics = metrics)
      
      
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
        SRE_train_rules_AUC <- fit_rules_SGL(inner_train_bake, drop_na(tibble(rules = RE_model$finalModel$rules$description))$rules)
        SRE_test_rules_AUC <- fit_rules_SGL(inner_test_bake, drop_na(tibble(rules = RE_model$finalModel$rules$description))$rules)
        
        SRE_train_AUC <- cbind(SRE_train_rules_AUC, fitted_smooths_train)
        SRE_test_AUC <- cbind(SRE_test_rules_AUC, fitted_smooths_test)
      } else {
        SRE_train_AUC <- cbind(inner_train_bake, fitted_smooths_train)
        SRE_test_AUC <- cbind(inner_test_bake, fitted_smooths_test)
      }
      if(!is.null(RE_model_Brier$finalModel$rules)) {
        SRE_train_rules_Brier <- fit_rules_SGL(inner_train_bake, drop_na(tibble(rules = RE_model_Brier$finalModel$rules$description))$rules)
        SRE_test_rules_Brier<- fit_rules_SGL(inner_test_bake, drop_na(tibble(rules = RE_model_Brier$finalModel$rules$description))$rules)
        
        SRE_train_Brier <- cbind(SRE_train_rules_Brier, fitted_smooths_train)
        SRE_test_Brier <- cbind(SRE_test_rules_Brier, fitted_smooths_test)
      } else {
        SRE_train_Brier <- cbind(inner_train_bake, fitted_smooths_train)
        SRE_test_Brier <- cbind(inner_test_bake, fitted_smooths_test)
      }
      if(!is.null(RE_model_PG$finalModel$rules)) {
        SRE_train_rules_PG <- fit_rules_SGL(inner_train_bake, drop_na(tibble(rules = RE_model_PG$finalModel$rules$description))$rules)
        SRE_test_rules_PG <- fit_rules_SGL(inner_test_bake, drop_na(tibble(rules = RE_model_PG$finalModel$rules$description))$rules)
        
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
      
      
      AUC_inner_SGL_table_train <- SRE_recipe_AUC %>%prep()%>% bake(training(SRE_split_AUC$splits[[1]]))
      Brier_inner_SGL_table_train <- SRE_recipe_Brier %>%prep()%>% bake(training(SRE_split_Brier$splits[[1]]))
      PG_inner_SGL_table_train <- SRE_recipe_PG %>%prep()%>% bake(training(SRE_split_PG$splits[[1]]))
      
      AUC_inner_SGL_table_test <- SRE_recipe_AUC %>%prep()%>% bake(assessment(SRE_split_AUC$splits[[1]]))
      Brier_inner_SGL_table_test <- SRE_recipe_Brier %>%prep()%>% bake(assessment(SRE_split_Brier$splits[[1]]))
      PG_inner_SGL_table_test <- SRE_recipe_PG %>%prep()%>% bake(assessment(SRE_split_PG$splits[[1]]))
      
      AUC_inner_SGL_table_train_x <- AUC_inner_SGL_table_train %>% select(-label)
      Brier_inner_SGL_table_train_x <- Brier_inner_SGL_table_train %>% select(-label)
      PG_inner_SGL_table_train_x <- PG_inner_SGL_table_train %>% select(-label)
      
      AUC_inner_groups <- group_terms_by_variables(names(AUC_inner_SGL_table_train_x), names(train))
      Brier_inner_groups <- group_terms_by_variables(names(Brier_inner_SGL_table_train_x), names(train))
      PG_inner_groups <- group_terms_by_variables(names(PG_inner_SGL_table_train_x), names(train))
      
      
      group_glm_AUC <- cv.sparsegl(as.matrix(AUC_inner_SGL_table_train_x)[, order(AUC_inner_groups)], AUC_inner_SGL_table_train$label, sort(AUC_inner_groups), family = "binomial")
      group_glm_Brier <- cv.sparsegl(as.matrix(Brier_inner_SGL_table_train_x)[, order(Brier_inner_groups)], Brier_inner_SGL_table_train$label, sort(Brier_inner_groups), family = "binomial")
      group_glm_PG <- cv.sparsegl(as.matrix(PG_inner_SGL_table_train_x)[, order(PG_inner_groups)], PG_inner_SGL_table_train$label, sort(PG_inner_groups), family = "binomial")
      
      
      lambdas_AUC <- c(lambdas_AUC, group_glm_AUC$lambda.min)
      lambdas_Brier <- c(lambdas_Brier, group_glm_Brier$lambda.min)
      lambdas_PG <- c(lambdas_PG, group_glm_PG$lambda.min)
      
      
    }
    
    print("hyperparameters found")
    
    ####### 
    
    best_lambda_auc <- mean(lambdas_AUC)
    best_lambda_Brier <- mean(lambdas_Brier)
    best_lambda_PG <- mean(lambdas_PG)
    
    
    
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
      SRE_train_rules_AUC <- fit_rules_SGL(train_bake, drop_na(tibble(rules = RE_model$finalModel$rules$description))$rules)
      SRE_test_rules_AUC <- fit_rules_SGL(test_bake, drop_na(tibble(rules = RE_model$finalModel$rules$description))$rules)
    } else {
      SRE_train_rules_AUC <- train_bake
      SRE_test_rules_AUC <- test_bake
    }
    
    if(!is.null(RE_model_Brier$finalModel$rules)) {
      SRE_train_rules_Brier <- fit_rules_SGL(train_bake, drop_na(tibble(rules = RE_model_Brier$finalModel$rules$description))$rules)
      SRE_test_rules_Brier <- fit_rules_SGL(test_bake, drop_na(tibble(rules = RE_model_Brier$finalModel$rules$description))$rules)
    } else {
      SRE_train_rules_Brier <- train_bake
      SRE_test_rules_Brier <- test_bake
    }
    
    if(!is.null(RE_model_PG$finalModel$rules)) {
      SRE_train_rules_PG <- fit_rules_SGL(train_bake, drop_na(tibble(rules = RE_model_PG$finalModel$rules$description))$rules)
      SRE_test_rules_PG <- fit_rules_SGL(test_bake, drop_na(tibble(rules = RE_model_PG$finalModel$rules$description))$rules)
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
    
    
    AUC_SGL_table_train <- SRE_recipe_AUC %>%prep()%>% bake(training(SRE_split_AUC$splits[[1]]))
    Brier_SGL_table_train <- SRE_recipe_Brier %>%prep()%>% bake(training(SRE_split_Brier$splits[[1]]))
    PG_SGL_table_train <- SRE_recipe_PG %>%prep()%>% bake(training(SRE_split_PG$splits[[1]]))
    
    AUC_SGL_table_test <- SRE_recipe_AUC %>%prep()%>% bake(assessment(SRE_split_AUC$splits[[1]]))
    Brier_SGL_table_test <- SRE_recipe_Brier %>%prep()%>% bake(assessment(SRE_split_Brier$splits[[1]]))
    PG_SGL_table_test <- SRE_recipe_PG %>%prep()%>% bake(assessment(SRE_split_PG$splits[[1]]))
    
    AUC_SGL_table_train_x <- AUC_SGL_table_train %>% select(-label)
    Brier_SGL_table_train_x <- Brier_SGL_table_train %>% select(-label)
    PG_SGL_table_train_x <- PG_SGL_table_train %>% select(-label)
    
    AUC_groups <- group_terms_by_variables(names(AUC_SGL_table_train_x), names(train))
    Brier_groups <- group_terms_by_variables(names(Brier_SGL_table_train_x), names(train))
    PG_groups <- group_terms_by_variables(names(PG_SGL_table_train_x), names(train))
    
    
    group_glm_AUC <- cv.sparsegl(as.matrix(AUC_SGL_table_train_x)[, order(AUC_groups)], AUC_SGL_table_train$label, sort(AUC_groups), family = "binomial")
    group_glm_Brier <- sparsegl(as.matrix(Brier_SGL_table_train_x)[, order(Brier_groups)], Brier_SGL_table_train$label, sort(Brier_groups), family = "binomial", lambda = best_lambda_Brier)
    group_glm_PG <- sparsegl(as.matrix(PG_SGL_table_train_x)[, order(PG_groups)], PG_SGL_table_train$label, sort(PG_groups), family = "binomial", lambda = best_lambda_PG)
    

    SRE_SGL_preds_AUC <- as.data.frame(predict(group_glm_AUC, as.matrix(AUC_SGL_table_test%>%select(-label)), type = 'response'))
    SRE_SGL_preds_Brier <- as.data.frame(predict(group_glm_Brier, as.matrix(Brier_SGL_table_test%>%select(-label)), type = 'response'))
    SRE_SGL_preds_PG <- as.data.frame(predict(group_glm_PG, as.matrix(PG_SGL_table_test%>%select(-label)), type = 'response'))
    
    names(SRE_SGL_preds_AUC) = names(SRE_SGL_preds_Brier) = names(SRE_SGL_preds_PG)="X1"
    
    SRE_SGL_preds_AUC$label <- test$label
    SRE_SGL_preds_Brier$label <- test$label
    SRE_SGL_preds_PG$label <- test$label
    
    g <- roc(label ~ X1, data = SRE_SGL_preds_AUC, direction = "<")
    AUC <- g$auc
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "RE", AUC)
    
    brier <- brier_score(SRE_SGL_preds_Brier$label, SRE_SGL_preds_Brier$X1)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "RE", brier)
    
    pg <- partialGini(SRE_SGL_preds_AUC$X1, SRE_SGL_preds_AUC$label)
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "RE", pg)
    
  }
  write.csv(AUC_results, file = paste("./results/",dataset_vector[dataset_counter],"_AUC.csv", sep = ""))
  write.csv(Brier_results, file = paste("./results/",dataset_vector[dataset_counter],"_BRIER.csv", sep = ""))
  PG_results$metric<-unlist(PG_results$metric)
  write.csv(PG_results, file = paste("./results/",dataset_vector[dataset_counter],"_PG.csv", sep = ""))
  
  dataset_counter <- dataset_counter + 1
}





##############TESTING ZONE

train_SGL <- SRE_recipe_AUC%>%prep(training(SRE_split_AUC$splits[[1]])) %>% bake(training(SRE_split_AUC$splits[[1]]))
fit_rules_SGL(train_SGL, RE_model$finalModel$rules$description)
names(SRE_train_rules_AUC)


group_terms_by_variables <- function(terms, original_vars) {
  # Initialize a list to hold the groups
  grouped_terms <- list()
  
  # Function to find the original variables in a term
  find_variables <- function(term, original_vars) {
    sapply(original_vars, function(var) grepl(paste0("(\\b|_)", var, "(\\b|_)"), term))
  }
  
  # Loop through each term
  for (term in terms) {
    # Find which original variables are in the term
    contains_vars <- original_vars[find_variables(term, original_vars)]
    
    # Convert to a sorted, comma-separated string to use as a list key
    key <- paste(sort(contains_vars), collapse = ",")
    
    # Add the term to the corresponding group
    if (key %in% names(grouped_terms)) {
      grouped_terms[[key]] <- c(grouped_terms[[key]], term)
    } else {
      grouped_terms[[key]] <- term
    }
  }
  
  names(grouped_terms) <- seq(1:length(grouped_terms))
  
  for (term_c in 1:length(terms)) {
    column_name <- terms[term_c]
    
    # Find the group that this column name belongs to
    group_name <- NULL
    for (group in names(grouped_terms)) {
      if (column_name %in% grouped_terms[[group]]) {
        group_name <- group
        break
      }
    }
    
    # Append the group name to the groups vector
    groups <- c(groups, group_name)
  }
  
  return(groups)
}


testgroups <- group_terms_by_variables(names(SRE_train_rules_AUC%>%select(-label)), names(train%>%select(-label)))
names(testgroups) <- seq(1:length(testgroups))
groups <- c()

for (term_c in 1:length(names(SRE_train_rules_AUC %>% select(-label)))) {
  column_name <- names(SRE_train_rules_AUC %>% select(-label))[term_c]
  
  # Find the group that this column name belongs to
  group_name <- NULL
  for (group in names(testgroups)) {
    if (column_name %in% testgroups[[group]]) {
      group_name <- group
      break
    }
  }
  
  # Append the group name to the groups vector
  groups <- c(groups, group_name)
}



vec <- as.numeric(testgroups)
x <- rbind(SRE_train_rules_AUC%>%select(-label), vec)

group_glm <- cv.sparsegl(as.matrix((SRE_train_rules_AUC%>%select(-label))[, order(vec)]), SRE_train_rules_AUC$label, sort(vec), family = "binomial")



###############





#write.csv(metric_results, file = "./results/AUCROC_results.csv")

stopCluster(cl)


set.seed(123)
train_indices <- sample(1:nrow(x), 0.8 * nrow(x))
x_train <- (x[train_indices, ])
x_test <- (x[-train_indices, ])
y_train <- (y[train_indices, ])
y_test <- (y[-train_indices, ])
