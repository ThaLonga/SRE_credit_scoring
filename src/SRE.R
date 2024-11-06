if (!require("pacman")) install.packages("pacman") ; require("pacman")
p_load(glmnet, glmnetUtils, mgcv, MASS, tidyverse, xgboost, DiagrammeR, stringr, tictoc, doParallel, pROC, earth, Matrix, pre, caret, parsnip, ggplot2, recipes, rsample, workflows, healthyR.ai, rlang, yardstick, bonsai, lightgbm, ranger, tune, DescTools, rules, discrim, party, partykit)
source("./src/hyperparameters.R")

cv.SRE <- function(inner_folds, tree_algorithm, RE_model_AUC, RE_model_Brier, RE_model_PG, RE_model_EMP, GAM_recipe, metrics, train_bake, test_bake) {
  stopifnot(is(inner_folds, "vfold_cv")||is(inner_folds, "rset"))
  stopifnot(is(metrics, "class_prob_metric_set"))
  stopifnot(is(tree_algorithm, "character"))
  stopifnot(is(GAM_recipe, "recipe"))
  stopifnot(is(train_bake, "data.frame"))
  stopifnot(is(test_bake, "data.frame"))
            
  full_metrics_AUC <- list()
  full_metrics_Brier <- list()
  full_metrics_PG <- list()
  full_metrics_EMP <- list()
  
  full_metrics_AUC_ridge <- list()
  full_metrics_Brier_ridge <- list()
  full_metrics_PG_ridge <- list()
  full_metrics_EMP_ridge <- list()
  
  
  for(k in 1:nrow(inner_folds)) {
    cat("hyperparameter search:\n")
    cat("SRE inner fold", k, "/", nrow(inner_folds),  "\n")
    ####### 
    # Data is split in training and test, preprocessing is applied
    
    inner_train <- analysis(inner_folds$splits[[k]])
    inner_test <- assessment(inner_folds$splits[[k]])
    
    inner_train_bake <- XGB_recipe %>% prep(inner_train) %>% bake(inner_train)
    inner_test_bake <- XGB_recipe %>% prep(inner_train) %>% bake(inner_test)
    
    #######
    # Rule ensembles
    if(identical(tree_algorithm,"bagging")) {
      cat("training RE")
      set.seed(k)
      RE_model_inner <- train(XGB_recipe, data = inner_train, method = "pre",
                            ntrees = 100, family = "binomial", trControl = trainControl(method = "none", classProbs = TRUE),
                            tuneGrid = preGrid_bag, ad.alpha = 0, tree.unbiased = FALSE, 
                            singleconditions = FALSE,
                            winsfrac = 0.05, normalize = TRUE, #same a priori influence as a typical rule
                            verbose = TRUE,
                            metric = "AUCROC", allowParallel = TRUE,
                            par.init=TRUE,
                            par.final=TRUE)
      
      RE_model_inner_AUC = RE_model_inner
      RE_model_inner_Brier = RE_model_inner
      RE_model_inner_PG = RE_model_inner
      RE_model_inner_EMP = RE_model_inner
      
    }
    else if(identical(tree_algorithm,"randomforest")) {
      cat("training RE")
      set.seed(k)
      RE_model_inner_AUC <- train(XGB_recipe, data = inner_train, method = "pre",
                           ntrees = 100, family = "binomial", trControl = ctrl,
                           tuneGrid = preGrid_RF, ad.alpha = 0, #tree.unbiased = FALSE, 
                           singleconditions = FALSE,
                           winsfrac = 0.05, normalize = TRUE, #same a priori influence as a typical rule
                           verbose = TRUE,
                           metric = "AUCROC", allowParallel = TRUE,
                           par.init=TRUE,
                           par.final=TRUE)   
      
      # fit on inner training set
#      RE_model_inner <- train(XGB_recipe, data = analysis(inner_split$splits[[1]]), method = "pre",
#                              ntrees = 100, family = "binomial", trControl = trainControl(method = "none", classProbs = TRUE),
#                              tuneGrid = RE_model_AUC_full$bestTune, ad.alpha = 0, #tree.unbiased = FALSE, 
#                              singleconditions = TRUE,
#                              winsfrac = 0.05, normalize = TRUE, #same a priori influence as a typical rule
#                              verbose = TRUE,
#                              metric = "AUCROC", allowParallel = TRUE,
#                              par.init=TRUE,
#                              par.final=TRUE)
      
      RE_model_inner_Brier <- train(XGB_recipe, data = train, method = "pre",
                                 ntrees = 100, family = "binomial", trControl = trainControl(method = "none", classProbs = TRUE),
                                 tuneGrid = getModelInfo("pre")[[1]]$grid( 
                                   maxdepth = (RE_model_inner_AUC$results%>%slice_min(Brier)%>%dplyr::select(maxdepth))[[1]][1],
                                   penalty.par.val = c("lambda.min"), # λand γ combination yielding the sparsest solution within 1 standard error of the error criterion of the minimum is returned
                                   sampfrac = 1,
                                   mtry = (RE_model_inner_AUC$results%>%slice_min(Brier)%>%dplyr::select(mtry))[[1]][1],
                                   use.grad = FALSE),
                                 ad.alpha = 0, #tree.unbiased = FALSE, 
                                 singleconditions = FALSE,
                                 winsfrac = 0.05, normalize = TRUE, #same a priori influence as a typical rule
                                 verbose = TRUE,
                                 allowParallel = TRUE,
                                 par.init=TRUE,
                                 par.final=TRUE)
      
      RE_model_inner_PG <- train(XGB_recipe, data = train, method = "pre",
                              ntrees = 100, family = "binomial", trControl = trainControl(method = "none", classProbs = TRUE),
                              tuneGrid = getModelInfo("pre")[[1]]$grid( 
                                maxdepth = (RE_model_inner_AUC$results%>%slice_max(partialGini)%>%dplyr::select(maxdepth))[[1]][1],
                                penalty.par.val = c("lambda.min"), # λand γ combination yielding the sparsest solution within 1 standard error of the error criterion of the minimum is returned
                                sampfrac = 1,
                                mtry = (RE_model_inner_AUC$results%>%slice_max(partialGini)%>%dplyr::select(mtry))[[1]][1],
                                use.grad = TRUE), ad.alpha = 0, #tree.unbiased = FALSE, 
                              singleconditions = FALSE,
                              winsfrac = 0.05, normalize = TRUE, #same a priori influence as a typical rule
                              verbose = TRUE,
                              metric = "AUCROC", allowParallel = TRUE,
                              par.init=TRUE,
                              par.final=TRUE)
      
      RE_model_inner_EMP <- train(XGB_recipe, data = train, method = "pre",
                                 ntrees = 100, family = "binomial", trControl = trainControl(method = "none", classProbs = TRUE),
                                 tuneGrid = getModelInfo("pre")[[1]]$grid( 
                                   maxdepth = (select_best_emp_RE_RF(RE_model_inner_AUC)%>%dplyr::select(maxdepth))[[1]][1],
                                   penalty.par.val = c("lambda.min"), # λand γ combination yielding the sparsest solution within 1 standard error of the error criterion of the minimum is returned
                                   sampfrac = 1,
                                   mtry = (select_best_emp_RE_RF(RE_model_inner_AUC)%>%dplyr::select(mtry))[[1]][1],
                                   use.grad = TRUE), ad.alpha = 0, #tree.unbiased = FALSE, 
                                 singleconditions = FALSE,
                                 winsfrac = 0.05, normalize = TRUE, #same a priori influence as a typical rule
                                 verbose = TRUE,
                                 metric = "AUCROC", allowParallel = TRUE,
                                 par.init=TRUE,
                                 par.final=TRUE)
      
      
      
    }
    else if(identical(tree_algorithm,"boosting")) {
      cat("training RE")
      set.seed(k)
      RE_model_inner_AUC <- train(XGB_recipe, data = inner_train, method = "pre",
                                 ntrees = 100, family = "binomial", trControl = ctrl,
                                 tuneGrid = preGrid_boosting, ad.alpha = 0,
                                 tree.unbiased = FALSE, singleconditions = TRUE,
                                 winsfrac = 0.05, normalize = TRUE, #same a priori influence as a typical rule
                                 verbose = TRUE,
                                 metric = "AUCROC", allowParallel = TRUE,
                                 par.init=TRUE,
                                 par.final=TRUE)
      # fit on inner training set
#      RE_model_inner <- train(XGB_recipe, data = analysis(inner_split$splits[[1]]), method = "pre",
#                              ntrees = 100, family = "binomial", trControl = trainControl(method = "none", classProbs = TRUE),
#                              tuneGrid = RE_model_AUC_full$bestTune, 
#                              tree.unbiased = FALSE, ad.alpha = 0, singleconditions = TRUE,
#                              winsfrac = 0.05, normalize = TRUE, #same a priori influence as a typical rule
#                              verbose = TRUE,
#                              metric = "AUCROC", allowParallel = TRUE,
#                              par.init=TRUE,
#                              par.final=TRUE)
      
      RE_model_inner_Brier <- train(XGB_recipe, data = inner_train, method = "pre",
                                       ntrees = 100, family = "binomial", trControl = trainControl(method = "none", classProbs = TRUE),
                                       tuneGrid = getModelInfo("pre")[[1]]$grid( 
                                         maxdepth = (RE_model_inner_AUC$results%>%slice_min(Brier)%>%dplyr::select(maxdepth))[[1]][1],
                                         learnrate = (RE_model_inner_AUC$results%>%slice_min(Brier)%>%dplyr::select(learnrate))[[1]][1],
                                         penalty.par.val = c("lambda.min"), # λand γ combination yielding the sparsest solution within 1 standard error of the error criterion of the minimum is returned
                                         sampfrac = 1,
                                         use.grad = TRUE), 
                                    tree.unbiased = FALSE, ad.alpha = 0, singleconditions = TRUE,
                                    winsfrac = 0.05, normalize = TRUE, #same a priori influence as a typical rule
                                    verbose = TRUE,
                                    allowParallel = TRUE,
                                    par.init=TRUE,
                                    par.final=TRUE)
      
      RE_model_inner_PG <- train(XGB_recipe, data = inner_train, method = "pre",
                                 ntrees = 100, family = "binomial", trControl = trainControl(method = "none", classProbs = TRUE),
                                 tuneGrid = getModelInfo("pre")[[1]]$grid( 
                                   maxdepth = (RE_model_inner_AUC$results%>%slice_max(partialGini)%>%dplyr::select(maxdepth))[[1]][1],
                                   learnrate = (RE_model_inner_AUC$results%>%slice_max(partialGini)%>%dplyr::select(learnrate))[[1]][1],
                                   penalty.par.val = c("lambda.1se"), # λand γ combination yielding the sparsest solution within 1 standard error of the error criterion of the minimum is returned
                                   sampfrac = 1,
                                   use.grad = TRUE), ad.alpha = 0, 
                                 tree.unbiased = FALSE, singleconditions = TRUE,
                                 winsfrac = 0.05, normalize = TRUE, #same a priori influence as a typical rule
                                 verbose = TRUE,
                                 metric = "AUCROC", allowParallel = TRUE,
                                 par.init=TRUE,
                                 par.final=TRUE)
      
      RE_model_inner_EMP <- train(XGB_recipe, data = inner_train, method = "pre",
                                  ntrees = 100, family = "binomial", trControl = trainControl(method = "none", classProbs = TRUE),
                                  tuneGrid = getModelInfo("pre")[[1]]$grid( 
                                    maxdepth = (select_best_emp_RE_boosting(RE_model_inner_AUC)%>%dplyr::select(maxdepth))[[1]][1],
                                    learnrate = (select_best_emp_RE_boosting(RE_model_inner_AUC)%>%dplyr::select(learnrate))[[1]][1],
                                    penalty.par.val = c("lambda.1se"), # λand γ combination yielding the sparsest solution within 1 standard error of the error criterion of the minimum is returned
                                    sampfrac = 1,
                                    use.grad = TRUE), ad.alpha = 0, 
                                  tree.unbiased = FALSE, singleconditions = TRUE,
                                  winsfrac = 0.05, normalize = TRUE, #same a priori influence as a typical rule
                                  verbose = TRUE,
                                  metric = "AUCROC", allowParallel = TRUE,
                                  par.init=TRUE,
                                  par.final=TRUE)      
    }
    else if(identical(tree_algorithm,"PLTR")) {
      cat("training RE")
      features <- setdiff(names(inner_train_bake), "label")  # Exclude the label column
      combinations <- combn(features, 2)  # Generate all combinations of 2 features
      rules = c()
      for (j in 1:ncol(combinations)) {
        feature_pair <- combinations[, j]
        formula <- as.formula(paste("label ~", paste(feature_pair, collapse = " + ")))
        tree <- as.party(rpart::rpart(formula, data = inner_train_bake, maxdepth = 2))
        extracted_rules <- partykit:::.list.rules.party(tree)
        if(extracted_rules[1]!= "") {rules <- c(rules, extracted_rules)}
      }
    }   
    
    ####### 
    # Fit GAM to extract splines only on numeric features with number of values >6
    cat("fitting splines\n")
    inner_train_gam_processed <- GAM_recipe%>%prep(inner_train)%>%bake(inner_train)
    
    smooth_vars = colnames(inner_train_gam_processed%>%dplyr::select(-label))[get_splineworthy_columns(inner_train_gam_processed)]
    inner_formula <- as.formula(
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
      add_model(GAM_model, formula = inner_formula)
    
    final_GAM_fit_inner <- GAM_wf %>% last_fit(inner_folds$splits[[k]], metrics = metrics)
    
    
    # Extract and fitted values for each smooth term
    smooth_terms <- grep("s\\(", unlist(str_split(as.character(inner_formula), " \\+ ")), value = TRUE)
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
    cat("fitting rules\n")
    if(identical(tree_algorithm, "PLTR")) {
      if(!is_empty(rules)) {
        SRE_train_rules_AUC <- fit_rules(inner_train_bake, unique(rules))
        SRE_test_rules_AUC <- fit_rules(inner_test_bake, unique(rules))
        SRE_train_rules_Brier <- SRE_train_rules_AUC
        SRE_test_rules_Brier <- SRE_test_rules_AUC
        SRE_train_rules_PG <- SRE_train_rules_AUC
        SRE_test_rules_PG <- SRE_test_rules_AUC
        SRE_train_rules_EMP <- SRE_train_rules_AUC
        SRE_test_rules_EMP <- SRE_test_rules_AUC
        
        SRE_train_AUC <- cbind(SRE_train_rules_AUC, fitted_smooths_train)
        SRE_test_AUC <- cbind(SRE_test_rules_AUC, fitted_smooths_test)
        SRE_train_Brier <- cbind(SRE_train_rules_AUC, fitted_smooths_train)
        SRE_test_Brier <- cbind(SRE_test_rules_AUC, fitted_smooths_test)
        SRE_train_PG <- cbind(SRE_train_rules_AUC, fitted_smooths_train)
        SRE_test_PG <- cbind(SRE_test_rules_AUC, fitted_smooths_test)
        SRE_train_EMP <- cbind(SRE_train_rules_AUC, fitted_smooths_train)
        SRE_test_EMP <- cbind(SRE_test_rules_AUC, fitted_smooths_test)
      } else {
        SRE_train_AUC <- cbind(inner_train_bake, fitted_smooths_train)
        SRE_test_AUC <- cbind(inner_test_bake, fitted_smooths_test)
        SRE_train_Brier <- cbind(inner_train_bake, fitted_smooths_train)
        SRE_test_Brier <- cbind(inner_test_bake, fitted_smooths_test)
        SRE_train_PG <- cbind(inner_train_bake, fitted_smooths_train)
        SRE_test_PG <- cbind(inner_test_bake, fitted_smooths_test)
        SRE_train_EMP <- cbind(inner_train_bake, fitted_smooths_train)
        SRE_test_EMP <- cbind(inner_test_bake, fitted_smooths_test)
      }
    }
    else {
      if(!is.null(RE_model_inner_AUC$finalModel$rules)) {
        SRE_train_rules_AUC <- fit_rules(inner_train_bake, drop_na(tibble(rules = RE_model_inner_AUC$finalModel$rules$description))$rules)
        SRE_test_rules_AUC <- fit_rules(inner_test_bake, drop_na(tibble(rules = RE_model_inner_AUC$finalModel$rules$description))$rules)
        
        SRE_train_AUC <- cbind(SRE_train_rules_AUC, fitted_smooths_train)
        SRE_test_AUC <- cbind(SRE_test_rules_AUC, fitted_smooths_test)
      } else {
        SRE_train_AUC <- cbind(inner_train_bake, fitted_smooths_train)
        SRE_test_AUC <- cbind(inner_test_bake, fitted_smooths_test)
      }
      if(!is.null(RE_model_inner_Brier$finalModel$rules)) {
        SRE_train_rules_Brier <- fit_rules(inner_train_bake, drop_na(tibble(rules = RE_model_inner_Brier$finalModel$rules$description))$rules)
        SRE_test_rules_Brier<- fit_rules(inner_test_bake, drop_na(tibble(rules = RE_model_inner_Brier$finalModel$rules$description))$rules)
        
        SRE_train_Brier <- cbind(SRE_train_rules_Brier, fitted_smooths_train)
        SRE_test_Brier <- cbind(SRE_test_rules_Brier, fitted_smooths_test)
      } else {
        SRE_train_Brier <- cbind(inner_train_bake, fitted_smooths_train)
        SRE_test_Brier <- cbind(inner_test_bake, fitted_smooths_test)
      }
      if(!is.null(RE_model_inner_PG$finalModel$rules)) {
        SRE_train_rules_PG <- fit_rules(inner_train_bake, drop_na(tibble(rules = RE_model_inner_PG$finalModel$rules$description))$rules)
        SRE_test_rules_PG <- fit_rules(inner_test_bake, drop_na(tibble(rules = RE_model_inner_PG$finalModel$rules$description))$rules)
        
        SRE_train_PG <- cbind(SRE_train_rules_PG, fitted_smooths_train)
        SRE_test_PG <- cbind(SRE_test_rules_PG, fitted_smooths_test)
      } else {
        SRE_train_PG <- cbind(inner_train_bake, fitted_smooths_train)
        SRE_test_PG <- cbind(inner_test_bake, fitted_smooths_test)
      }
      if(!is.null(RE_model_inner_EMP$finalModel$rules)) {
        SRE_train_rules_EMP <- fit_rules(inner_train_bake, drop_na(tibble(rules = RE_model_inner_EMP$finalModel$rules$description))$rules)
        SRE_test_rules_EMP <- fit_rules(inner_test_bake, drop_na(tibble(rules = RE_model_inner_EMP$finalModel$rules$description))$rules)
        
        SRE_train_EMP <- cbind(SRE_train_rules_EMP, fitted_smooths_train)
        SRE_test_EMP <- cbind(SRE_test_rules_EMP, fitted_smooths_test)
      } else {
        SRE_train_EMP <- cbind(inner_train_bake, fitted_smooths_train)
        SRE_test_EMP <- cbind(inner_test_bake, fitted_smooths_test)
      }
    }
    
    ####### 
    # Again, only winsorize numeric features with more than 6 values
    winsorizable_AUC <- get_splineworthy_columns(SRE_train_AUC)
    winsorizable_Brier <- get_splineworthy_columns(SRE_train_Brier)
    winsorizable_PG <- get_splineworthy_columns(SRE_train_PG)
    winsorizable_EMP <- get_splineworthy_columns(SRE_train_EMP)
    
    #######
    # Manually create Rsample splits again, now with the splines and rules added
    indices <- list(
      list(analysis = 1:nrow(SRE_train_AUC), assessment = (nrow(SRE_train_AUC)+1):(nrow(SRE_train_AUC)+nrow(SRE_test_AUC)))
    )
    splits_AUC <- lapply(indices, make_splits, data = rbind(SRE_train_AUC, SRE_test_AUC))
    splits_Brier <- lapply(indices, make_splits, data = rbind(SRE_train_Brier, SRE_test_Brier))
    splits_PG <- lapply(indices, make_splits, data = rbind(SRE_train_PG, SRE_test_PG))
    splits_EMP <- lapply(indices, make_splits, data = rbind(SRE_train_EMP, SRE_test_EMP))
    SRE_split_AUC <- manual_rset(splits_AUC, c("Split SRE"))
    SRE_split_Brier <- manual_rset(splits_Brier, c("Split SRE"))
    SRE_split_PG <- manual_rset(splits_PG, c("Split SRE"))
    SRE_split_EMP <- manual_rset(splits_EMP, c("Split SRE"))
    
    ####### 
    # Create recipes, for AUC, Brier and PG
    normalizable_AUC <- colnames(training(SRE_split_AUC$splits[[1]])[unlist(lapply(training(SRE_split_AUC$splits[[1]]), function(x) n_distinct(x)>2))])
    normalizable_Brier <- colnames(training(SRE_split_Brier$splits[[1]])[unlist(lapply(training(SRE_split_Brier$splits[[1]]), function(x) n_distinct(x)>2))])
    normalizable_PG <- colnames(training(SRE_split_PG$splits[[1]])[unlist(lapply(training(SRE_split_PG$splits[[1]]), function(x) n_distinct(x)>2))])
    normalizable_EMP <- colnames(training(SRE_split_EMP$splits[[1]])[unlist(lapply(training(SRE_split_EMP$splits[[1]]), function(x) n_distinct(x)>2))])
    
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
    
    SRE_recipe_EMP <- recipe(label~., data = training(SRE_split_EMP$splits[[1]])) %>%
      step_hai_winsorized_truncate(all_of(names(!!training(SRE_split_EMP$splits[[1]]))[!!winsorizable_EMP]), fraction = 0.025) %>%
      step_rm(all_of(names(!!training(SRE_split_EMP$splits[[1]]))[!!winsorizable_EMP])) %>%
      step_mutate_at(contains("winsorized"), fn = ~0.4 * ./ sd(.)) %>%
      step_mutate(across(where(is.logical), as.integer)) %>%
      step_normalize(all_of(setdiff(!!normalizable_EMP, colnames(!!training(SRE_split_EMP$splits[[1]])[!!winsorizable_EMP])))) %>%
      step_zv()
    
    
    #######
    # Ridge for alasso
    cat("estimating ridge coefficients\n")
    ridge_model <- 
      parsnip::logistic_reg(
        mode = "classification",
        mixture = 0,
        penalty = tune()
      ) %>%
      set_engine("glmnet")
    
    SRE_ridge_wf_AUC <- workflow() %>%
      add_recipe(SRE_recipe_AUC) %>%
      add_model(ridge_model)
    SRE_ridge_wf_Brier <- workflow() %>%
      add_recipe(SRE_recipe_Brier) %>%
      add_model(ridge_model)
    SRE_ridge_wf_PG <- workflow() %>%
      add_recipe(SRE_recipe_PG) %>%
      add_model(ridge_model)
    SRE_ridge_wf_EMP <- workflow() %>%
      add_recipe(SRE_recipe_EMP) %>%
      add_model(ridge_model)
    
    SRE_ridge_tuned_AUC <- tune::tune_grid(
      object = SRE_ridge_wf_AUC,
      resamples = SRE_split_AUC,
      grid = hyperparameters_SRE_tidy, 
      metrics = metrics,
      control = tune::control_grid(verbose = TRUE, save_pred = TRUE, parallel_over = "everything"))
    SRE_ridge_tuned_Brier <- tune::tune_grid(
      object = SRE_ridge_wf_Brier,
      resamples = SRE_split_Brier,
      grid = hyperparameters_SRE_tidy, 
      metrics = metrics,
      control = tune::control_grid(verbose = TRUE, save_pred = TRUE, parallel_over = "everything"))
    SRE_ridge_tuned_PG <- tune::tune_grid(
      object = SRE_ridge_wf_PG,
      resamples = SRE_split_PG,
      grid = hyperparameters_SRE_tidy, 
      metrics = metrics,
      control = tune::control_grid(verbose = TRUE, save_pred = TRUE, parallel_over = "everything"))
    SRE_ridge_tuned_EMP <- tune::tune_grid(
      object = SRE_ridge_wf_EMP,
      resamples = SRE_split_EMP,
      grid = hyperparameters_SRE_tidy, 
      metrics = metrics,
      control = tune::control_grid(verbose = TRUE, save_pred = TRUE, parallel_over = "everything"))

    #for auc, brier
    metrics_SRE_ridge_AUC <- SRE_ridge_tuned_AUC$.metrics[[1]]
    metrics_SRE_ridge_Brier <- SRE_ridge_tuned_Brier$.metrics[[1]]
    metrics_SRE_ridge_AUC$fold <- rep(k, nrow(metrics_SRE_ridge_AUC))
    metrics_SRE_ridge_Brier$fold <- rep(k, nrow(metrics_SRE_ridge_Brier))
    
    full_metrics_AUC_ridge <- rbind(full_metrics_AUC_ridge, metrics_SRE_ridge_AUC)
    full_metrics_Brier_ridge <- rbind(full_metrics_Brier_ridge, metrics_SRE_ridge_Brier)
    
    #for pg
    metrics_SRE_ridge_PG <- suppressMessages(SRE_ridge_tuned_PG%>%collect_predictions(summarize = FALSE) %>%
                                               group_by(id, penalty, .config) %>%
                                               summarise(partial_gini = partialGini(.pred_X1, label)))
    metrics_SRE_ridge_PG$fold <- rep(k, nrow(metrics_SRE_ridge_PG))
    
    full_metrics_PG_ridge <- rbind(full_metrics_PG_ridge, metrics_SRE_ridge_PG)

    #for emp
    metrics_SRE_ridge_EMP <- suppressMessages(SRE_ridge_tuned_EMP%>%collect_predictions(summarize = FALSE) %>%
                                               group_by(id, penalty, .config) %>%
                                               summarise(emp = empCreditScoring(.pred_X1, label)$EMPC))
    metrics_SRE_ridge_EMP$fold <- rep(k, nrow(metrics_SRE_ridge_EMP))
    
    full_metrics_EMP_ridge <- rbind(full_metrics_EMP_ridge, metrics_SRE_ridge_EMP)
    
    best_ridge_auc <- SRE_ridge_tuned_AUC %>% select_best(metric="roc_auc")
    final_ridge_wf_auc <- SRE_ridge_wf_AUC %>% finalize_workflow(best_ridge_auc)
    final_ridge_fit_auc <- final_ridge_wf_auc %>% last_fit(SRE_split_AUC$splits[[k]], metrics = metrics)
    ridge_penalties_AUC <- coef((final_ridge_fit_auc%>%extract_fit_engine()), s=best_ridge_auc$penalty)
    
    best_ridge_Brier <- SRE_ridge_tuned_Brier %>% select_best(metric="brier_class")
    final_ridge_wf_Brier <- SRE_ridge_wf_Brier %>% finalize_workflow(best_ridge_Brier)
    final_ridge_fit_Brier <- final_ridge_wf_Brier %>% last_fit(SRE_split_Brier$splits[[k]], metrics = metrics)
    ridge_penalties_Brier <- coef((final_ridge_fit_Brier%>%extract_fit_engine()), s=best_ridge_Brier$penalty)
    
    best_ridge_PG <- SRE_ridge_tuned_PG %>% select_best_pg_SRE()
    final_ridge_wf_PG <- SRE_ridge_wf_PG %>% finalize_workflow(best_ridge_PG)
    final_ridge_fit_PG <- final_ridge_wf_PG %>% last_fit(SRE_split_PG$splits[[k]], metrics = metrics)
    ridge_penalties_PG <- coef((final_ridge_fit_PG%>%extract_fit_engine()), s=best_ridge_PG$penalty)

    best_ridge_EMP <- SRE_ridge_tuned_EMP %>% select_best_emp_SRE()
    final_ridge_wf_EMP <- SRE_ridge_wf_EMP %>% finalize_workflow(best_ridge_EMP)
    final_ridge_fit_EMP <- final_ridge_wf_EMP %>% last_fit(SRE_split_EMP$splits[[k]], metrics = metrics)
    ridge_penalties_EMP <- coef((final_ridge_fit_EMP%>%extract_fit_engine()), s=best_ridge_EMP$penalty)
    
    
    
    ####### 
    # Fit regular lasso for AUC, Brier, PG
    cat("fitting adaptive lasso\n")
    
    SRE_model_AUC <- 
      parsnip::logistic_reg(
        mode = "classification",
        mixture = 1,
        penalty = tune()
      ) %>%
      set_engine("glmnet", penalty.factors = 1/abs(ridge_penalties_AUC[-1]))
    
    SRE_model_Brier <- 
      parsnip::logistic_reg(
        mode = "classification",
        mixture = 1,
        penalty = tune()
      ) %>%
      set_engine("glmnet", penalty.factors = 1/abs(ridge_penalties_Brier[-1]))
    
    SRE_model_PG <- 
      parsnip::logistic_reg(
        mode = "classification",
        mixture = 1,
        penalty = tune()
      ) %>%
      set_engine("glmnet", penalty.factors = 1/abs(ridge_penalties_PG[-1]))

    SRE_model_EMP <- 
      parsnip::logistic_reg(
        mode = "classification",
        mixture = 1,
        penalty = tune()
      ) %>%
      set_engine("glmnet", penalty.factors = 1/abs(ridge_penalties_EMP[-1]))
    
    SRE_wf_AUC <- workflow() %>%
      add_recipe(SRE_recipe_AUC) %>%
      add_model(SRE_model_AUC)
    SRE_wf_Brier <- workflow() %>%
      add_recipe(SRE_recipe_Brier) %>%
      add_model(SRE_model_Brier)
    SRE_wf_PG <- workflow() %>%
      add_recipe(SRE_recipe_PG) %>%
      add_model(SRE_model_PG)
    SRE_wf_EMP <- workflow() %>%
      add_recipe(SRE_recipe_EMP) %>%
      add_model(SRE_model_EMP)
    
    tic()
    SRE_tuned_AUC <- tune::tune_grid(
      object = SRE_wf_AUC,
      resamples = SRE_split_AUC,
      grid = hyperparameters_SRE_tidy, 
      metrics = metrics,
      control = tune::control_grid(verbose = TRUE, save_pred = TRUE, parallel_over = "everything"))
    SRE_tuned_Brier <- tune::tune_grid(
      object = SRE_wf_Brier,
      resamples = SRE_split_Brier,
      grid = hyperparameters_SRE_tidy, 
      metrics = metrics,
      control = tune::control_grid(verbose = TRUE, save_pred = TRUE, parallel_over = "everything"))
    SRE_tuned_PG <- tune::tune_grid(
      object = SRE_wf_PG,
      resamples = SRE_split_PG,
      grid = hyperparameters_SRE_tidy, 
      metrics = metrics,
      control = tune::control_grid(verbose = TRUE, save_pred = TRUE, parallel_over = "everything"))
    SRE_tuned_EMP <- tune::tune_grid(
      object = SRE_wf_EMP,
      resamples = SRE_split_EMP,
      grid = hyperparameters_SRE_tidy, 
      metrics = metrics,
      control = tune::control_grid(verbose = TRUE, save_pred = TRUE, parallel_over = "everything"))
    toc()
    
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
    
    #for emp
    metrics_SRE_EMP <- suppressMessages(SRE_tuned_EMP%>%collect_predictions(summarize = FALSE) %>%
                                         group_by(id, penalty, .config) %>%
                                         summarise(emp = empCreditScoring(.pred_X1, label)$EMPC))
    metrics_SRE_EMP$fold <- rep(k, nrow(metrics_SRE_EMP))
    
    full_metrics_EMP <- rbind(full_metrics_EMP, metrics_SRE_EMP)
  }
  
  cat("hyperparameters found")
  
  ####### 
  # Hyperparameter extraction, we use lambda.1se  = lambda.min + 1se
  
  # Ridge
  aggregated_metrics_ridge_AUC <- full_metrics_AUC_ridge %>% group_by(penalty, .config, .metric) %>%
    summarise(mean_perf = mean(.estimate))
  aggregated_metrics_ridge_Brier <- full_metrics_Brier_ridge %>% group_by(penalty, .config, .metric) %>%
    summarise(mean_perf = mean(.estimate))
  aggregated_metrics_ridge_PG <- full_metrics_PG_ridge %>% group_by(penalty, .config) %>%
    summarise(mean_perf = mean(partial_gini))
  aggregated_metrics_ridge_EMP <- full_metrics_EMP_ridge %>% group_by(penalty, .config) %>%
    summarise(mean_perf = mean(emp))
  
  best_lambda_ridge_auc <- aggregated_metrics_ridge_AUC %>% filter(.metric=="roc_auc") %>% ungroup() %>% slice_max(mean_perf) %>% slice_head() %>% dplyr::select(penalty) %>% pull()
  best_lambda_ridge_brier <- aggregated_metrics_ridge_Brier %>% filter(.metric=="brier_class") %>% ungroup() %>% slice_min(mean_perf) %>% slice_head() %>% dplyr::select(penalty) %>% pull()
  best_lambda_ridge_pg <- aggregated_metrics_ridge_PG %>% ungroup() %>% slice_max(mean_perf) %>% slice_head() %>% dplyr::select(penalty) %>% pull()
  best_lambda_ridge_emp <- aggregated_metrics_ridge_EMP %>% ungroup() %>% slice_max(mean_perf) %>% slice_head() %>% dplyr::select(penalty) %>% pull()
  
  
  # Lasso
  aggregated_metrics_AUC <- full_metrics_AUC %>% group_by(penalty, .config, .metric) %>%
    summarise(mean_perf = mean(.estimate))
  aggregated_metrics_Brier <- full_metrics_Brier %>% group_by(penalty, .config, .metric) %>%
    summarise(mean_perf = mean(.estimate))
  aggregated_metrics_PG <- full_metrics_PG %>% group_by(penalty, .config) %>%
    summarise(mean_perf = mean(partial_gini))
  aggregated_metrics_EMP <- full_metrics_EMP %>% group_by(penalty, .config) %>%
    summarise(mean_perf = mean(emp))
  
  best_lambda_auc <- aggregated_metrics_AUC %>% filter(.metric=="roc_auc") %>% ungroup() %>% slice_max(mean_perf) %>% slice_head() %>% dplyr::select(penalty) %>% pull()
  best_lambda_brier <- aggregated_metrics_Brier %>% filter(.metric=="brier_class") %>% ungroup() %>% slice_min(mean_perf) %>% slice_head() %>% dplyr::select(penalty) %>% pull()
  best_lambda_pg <- aggregated_metrics_PG %>% ungroup() %>% slice_max(mean_perf) %>% slice_head() %>% dplyr::select(penalty) %>% pull()
  best_lambda_emp <- aggregated_metrics_EMP %>% ungroup() %>% slice_max(mean_perf) %>% slice_head() %>% dplyr::select(penalty) %>% pull()
  
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
  
  lambda_sd_emp <- sd(unlist(
    full_metrics_EMP %>% 
      group_by(fold) %>%
      slice_max(emp) %>%
      slice_head() %>%
      ungroup() %>%
      dplyr::select(penalty)    
  ))
  
  
  
  lambda_1se_auc <- best_lambda_auc + lambda_sd_auc
  lambda_1se_brier <- best_lambda_brier + lambda_sd_brier
  lambda_1se_pg <- best_lambda_pg + lambda_sd_pg
  lambda_1se_emp <- best_lambda_emp + lambda_sd_emp
  
  
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
  if(identical(tree_algorithm, "PLTR")) {
    cat("training RE")
    features <- setdiff(names(train_bake), "label")  # Exclude the label column
    combinations <- combn(features, 2)  # Generate all combinations of 2 features
    rules = c()
    for (j in 1:ncol(combinations)) {
      feature_pair <- combinations[, j]
      formula <- as.formula(paste("label ~", paste(feature_pair, collapse = " + ")))
      tree <- as.party(rpart::rpart(formula, data = train_bake, maxdepth = 2))
      extracted_rules <- partykit:::.list.rules.party(tree)
      if(extracted_rules[1]!= "") {rules <- c(rules, extracted_rules)}
    }   
    if(!is_empty(rules)) {
      SRE_train_rules_AUC <- fit_rules(train_bake, unique(rules))
      SRE_test_rules_AUC <- fit_rules(test_bake, unique(rules))
      SRE_train_rules_Brier <- SRE_train_rules_AUC
      SRE_test_rules_Brier <- SRE_test_rules_AUC
      SRE_train_rules_PG <- SRE_train_rules_AUC
      SRE_test_rules_PG <- SRE_test_rules_AUC
      SRE_train_rules_EMP <- SRE_train_rules_AUC
      SRE_test_rules_EMP <- SRE_test_rules_AUC
      
      SRE_train_AUC <- cbind(SRE_train_rules_AUC, fitted_smooths_train)
      SRE_test_AUC <- cbind(SRE_test_rules_AUC, fitted_smooths_test)
      SRE_train_Brier <- cbind(SRE_train_rules_AUC, fitted_smooths_train)
      SRE_test_Brier <- cbind(SRE_test_rules_AUC, fitted_smooths_test)
      SRE_train_PG <- cbind(SRE_train_rules_AUC, fitted_smooths_train)
      SRE_test_PG <- cbind(SRE_test_rules_AUC, fitted_smooths_test)
      SRE_train_EMP <- cbind(SRE_train_rules_AUC, fitted_smooths_train)
      SRE_test_EMP <- cbind(SRE_test_rules_AUC, fitted_smooths_test)
    } else {
      SRE_train_AUC <- cbind(train_bake, fitted_smooths_train)
      SRE_test_AUC <- cbind(test_bake, fitted_smooths_test)
      SRE_train_Brier <- cbind(train_bake, fitted_smooths_train)
      SRE_test_Brier <- cbind(test_bake, fitted_smooths_test)
      SRE_train_PG <- cbind(train_bake, fitted_smooths_train)
      SRE_test_PG <- cbind(test_bake, fitted_smooths_test)
      SRE_train_EMP <- cbind(train_bake, fitted_smooths_train)
      SRE_test_EMP <- cbind(test_bake, fitted_smooths_test)
    }
  } else {
    if(!is.null(RE_model_AUC$finalModel$rules)) {
      SRE_train_rules_AUC <- fit_rules(train_bake, drop_na(tibble(rules = RE_model_AUC$finalModel$rules$description))$rules)
      SRE_test_rules_AUC <- fit_rules(test_bake, drop_na(tibble(rules = RE_model_AUC$finalModel$rules$description))$rules)
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
    
    if(!is.null(RE_model_EMP$finalModel$rules)) {
      SRE_train_rules_EMP <- fit_rules(train_bake, drop_na(tibble(rules = RE_model_EMP$finalModel$rules$description))$rules)
      SRE_test_rules_EMP <- fit_rules(test_bake, drop_na(tibble(rules = RE_model_EMP$finalModel$rules$description))$rules)
    } else {
      SRE_train_rules_EMP <- train_bake
      SRE_test_rules_EMP <- test_bake
    }
  }
  
  
  SRE_train_AUC <- cbind(SRE_train_rules_AUC, fitted_smooths_train)
  SRE_train_Brier <- cbind(SRE_train_rules_Brier, fitted_smooths_train)
  SRE_train_PG <- cbind(SRE_train_rules_PG, fitted_smooths_train)
  SRE_train_EMP <- cbind(SRE_train_rules_EMP, fitted_smooths_train)
  
  SRE_test_AUC <- cbind(SRE_test_rules_AUC, fitted_smooths_test)
  SRE_test_Brier <- cbind(SRE_test_rules_Brier, fitted_smooths_test)
  SRE_test_PG <- cbind(SRE_test_rules_PG, fitted_smooths_test)
  SRE_test_EMP <- cbind(SRE_test_rules_EMP, fitted_smooths_test)
  
  ####### 
  # Only winsorize numeric features with more than 6 values
  winsorizable_AUC <- get_splineworthy_columns(SRE_train_AUC)
  winsorizable_Brier <- get_splineworthy_columns(SRE_train_Brier)
  winsorizable_PG <- get_splineworthy_columns(SRE_train_PG)
  winsorizable_EMP <- get_splineworthy_columns(SRE_train_EMP)
  
  #######
  # Manually create Rsample splits again, now with the splines and rules added
  indices <- list(
    list(analysis = 1:nrow(SRE_train_AUC), assessment = (nrow(SRE_train_AUC)+1):(nrow(SRE_train_AUC)+nrow(SRE_test_AUC)))
  )
  
  splits_AUC <- lapply(indices, make_splits, data = rbind(SRE_train_AUC, SRE_test_AUC))
  splits_Brier <- lapply(indices, make_splits, data = rbind(SRE_train_Brier, SRE_test_Brier))
  splits_PG <- lapply(indices, make_splits, data = rbind(SRE_train_PG, SRE_test_PG))
  splits_EMP <- lapply(indices, make_splits, data = rbind(SRE_train_EMP, SRE_test_EMP))
  
  SRE_split_AUC <- manual_rset(splits_AUC, c("Split SRE"))
  SRE_split_Brier <- manual_rset(splits_Brier, c("Split SRE"))
  SRE_split_PG <- manual_rset(splits_PG, c("Split SRE"))
  SRE_split_EMP <- manual_rset(splits_EMP, c("Split SRE"))
  
  ####### 
  # Create recipes, for AUC, Brier and PG
  normalizable_AUC <- colnames(training(SRE_split_AUC$splits[[1]])[unlist(lapply(training(SRE_split_AUC$splits[[1]]), function(x) n_distinct(x)>2))])
  normalizable_Brier <- colnames(training(SRE_split_Brier$splits[[1]])[unlist(lapply(training(SRE_split_Brier$splits[[1]]), function(x) n_distinct(x)>2))])
  normalizable_PG <- colnames(training(SRE_split_PG$splits[[1]])[unlist(lapply(training(SRE_split_PG$splits[[1]]), function(x) n_distinct(x)>2))])
  normalizable_EMP <- colnames(training(SRE_split_EMP$splits[[1]])[unlist(lapply(training(SRE_split_EMP$splits[[1]]), function(x) n_distinct(x)>2))])
  
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
  
  SRE_recipe_EMP <- recipe(label~., data = training(SRE_split_EMP$splits[[1]])) %>%
    step_hai_winsorized_truncate(all_of(names(!!training(SRE_split_EMP$splits[[1]]))[!!winsorizable_EMP]), fraction = 0.025) %>%
    step_rm(all_of(names(!!training(SRE_split_EMP$splits[[1]]))[!!winsorizable_EMP])) %>%
    step_mutate_at(contains("winsorized"), fn = ~0.4 * ./ sd(.)) %>%
    step_mutate(across(where(is.logical), as.integer)) %>%
    step_normalize(all_of(setdiff(!!normalizable_EMP, colnames(!!training(SRE_split_EMP$splits[[1]])[!!winsorizable_EMP])))) %>%
    step_zv()
  
  
  #######
  # Fit initial ridge for alasso
  SRE_model_ridge_auc <- 
    parsnip::logistic_reg(
      mode = "classification",
      mixture = 0,
      penalty = best_lambda_ridge_auc
    ) %>%
    set_engine("glmnet")
  
  SRE_wf_ridge_auc <- workflow() %>%
    #add_formula(label~.) %>%
    add_recipe(SRE_recipe_AUC) %>%
    add_model(SRE_model_ridge_auc)
  
  SRE_model_ridge_brier <- 
    parsnip::logistic_reg(
      mode = "classification",
      mixture = 0,
      penalty = best_lambda_ridge_brier
    ) %>%
    set_engine("glmnet")
  
  SRE_wf_ridge_brier <- workflow() %>%
    #add_formula(label~.) %>%
    add_recipe(SRE_recipe_Brier) %>%
    add_model(SRE_model_ridge_brier)    
  
  SRE_model_ridge_pg <- 
    parsnip::logistic_reg(
      mode = "classification",
      mixture = 0,
      penalty = best_lambda_ridge_pg    #sd can be very high for PG resulting in way too high lambda, leaving only the intercept
    ) %>%
    set_engine("glmnet")
  
  SRE_wf_ridge_pg <- workflow() %>%
    #add_formula(label~.) %>%
    add_recipe(SRE_recipe_PG) %>%
    add_model(SRE_model_ridge_pg)    
  
  SRE_model_ridge_emp <- 
    parsnip::logistic_reg(
      mode = "classification",
      mixture = 0,
      penalty = best_lambda_ridge_emp    #sd can be very high for EMP resulting in way too high lambda, leaving only the intercept
    ) %>%
    set_engine("glmnet")
  
  SRE_wf_ridge_emp <- workflow() %>%
    #add_formula(label~.) %>%
    add_recipe(SRE_recipe_EMP) %>%
    add_model(SRE_model_ridge_emp)    
  
  
  final_ridge_fit_auc <- SRE_wf_ridge_auc %>% last_fit(SRE_split_AUC$splits[[1]], metrics = metrics)
  final_ridge_fit_brier <- SRE_wf_ridge_brier %>% last_fit(SRE_split_Brier$splits[[1]], metrics = metrics)
  final_ridge_fit_pg <- SRE_wf_ridge_pg %>% last_fit(SRE_split_PG$splits[[1]], metrics = metrics)
  final_ridge_fit_emp <- SRE_wf_ridge_emp %>% last_fit(SRE_split_EMP$splits[[1]], metrics = metrics)
  
  ridge_penalties_AUC <- coef((final_ridge_fit_auc%>%extract_fit_engine()), s=best_lambda_ridge_auc)
  ridge_penalties_Brier <- coef((final_ridge_fit_Brier%>%extract_fit_engine()), s=best_lambda_ridge_brier)
  ridge_penalties_PG <- coef((final_ridge_fit_PG%>%extract_fit_engine()), s=best_lambda_ridge_pg)
  ridge_penalties_EMP <- coef((final_ridge_fit_EMP%>%extract_fit_engine()), s=best_lambda_ridge_emp)
  
  
  ####### 
  # Fit regular lasso for AUC, Brier, PG and extract metrics
  SRE_model_auc <- 
    parsnip::logistic_reg(
      mode = "classification",
      mixture = 1,
      penalty = best_lambda_auc
    ) %>%
    set_engine("glmnet", penalty.factors = 1 / abs(ridge_penalties_AUC[-1]))
  
  SRE_wf_auc <- workflow() %>%
    #add_formula(label~.) %>%
    add_recipe(SRE_recipe_AUC) %>%
    add_model(SRE_model_auc)
  
  SRE_model_brier <- 
    parsnip::logistic_reg(
      mode = "classification",
      mixture = 1,
      penalty = best_lambda_brier
    ) %>%
    set_engine("glmnet", penalty.factors = 1 / abs(ridge_penalties_Brier[-1]))
  
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
    set_engine("glmnet", penalty.factors = 1 / abs(ridge_penalties_PG[-1]))
  
  SRE_wf_pg <- workflow() %>%
    #add_formula(label~.) %>%
    add_recipe(SRE_recipe_PG) %>%
    add_model(SRE_model_pg)   
  
  SRE_model_emp <- 
    parsnip::logistic_reg(
      mode = "classification",
      mixture = 1,
      penalty = best_lambda_emp    #sd can be very high for EMP resulting in way too high lambda, leaving only the intercept
    ) %>%
    set_engine("glmnet", penalty.factors = 1 / abs(ridge_penalties_EMP[-1]))
  
  SRE_wf_emp <- workflow() %>%
    #add_formula(label~.) %>%
    add_recipe(SRE_recipe_EMP) %>%
    add_model(SRE_model_emp)    
  
  
  final_SRE_fit_auc <- SRE_wf_auc %>% last_fit(SRE_split_AUC$splits[[1]], metrics = metrics)
  final_SRE_fit_brier <- SRE_wf_brier %>% last_fit(SRE_split_Brier$splits[[1]], metrics = metrics)
  final_SRE_fit_pg <- SRE_wf_pg %>% last_fit(SRE_split_PG$splits[[1]], metrics = metrics)
  final_SRE_fit_emp <- SRE_wf_emp %>% last_fit(SRE_split_EMP$splits[[1]], metrics = metrics)
  
  return(list(best_AUC = final_SRE_fit_auc, best_Brier = final_SRE_fit_brier, best_PG = final_SRE_fit_pg, best_EMP = final_SRE_fit_emp))
}
