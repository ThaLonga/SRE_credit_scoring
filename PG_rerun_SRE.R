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

dataset_counter = 4

for(dataset in datasets) {
  
  AUC_results <- metric_results
  Brier_results <- metric_results
  PG_results <- metric_results
  
  
  # for GMSC only 3 repeats because large dataset
  #if(dataset_counter==3) {nr_repeats <- 3} else {nr_repeats <- 5}
  nr_innerfolds = nr_repeats
  
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
    source("./src/hyperparameters.R")
    
    
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
      winsorizable_PG <- get_splineworthy_columns(SRE_train_PG)
      
      #######
      # Manually create Rsample splits again, now with the splines and rules added
      indices <- list(
        list(analysis = 1:nrow(SRE_train_PG), assessment = (nrow(SRE_train_PG)+1):(nrow(SRE_train_PG)+nrow(SRE_test_PG)))
      )
      splits_PG <- lapply(indices, make_splits, data = rbind(SRE_train_PG, SRE_test_PG))
      SRE_split_PG <- manual_rset(splits_PG, c("Split SRE"))
      
      ####### 
      # Create recipes, for AUC, Brier and PG
      normalizable_PG <- colnames(training(SRE_split_PG$splits[[1]])[unlist(lapply(training(SRE_split_PG$splits[[1]]), function(x) n_distinct(x)>2))])
      
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
      
      SRE_wf_PG <- workflow() %>%
        add_recipe(SRE_recipe_PG) %>%
        add_model(SRE_model)

      SRE_tuned_PG <- tune::tune_grid(
        object = SRE_wf_PG,
        resamples = SRE_split_PG,
        grid = hyperparameters_SRE_tidy, 
        metrics = metrics,
        control = tune::control_grid(verbose = TRUE, save_pred = TRUE))
      
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
    aggregated_metrics_PG <- full_metrics_PG %>% group_by(penalty, .config) %>%
      summarise(mean_perf = mean(partial_gini))
    
    best_lambda_pg <- aggregated_metrics_PG %>% ungroup() %>% slice_max(mean_perf) %>% slice_head() %>% dplyr::select(penalty) %>% pull()
    
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
    
    if(!is.null(RE_model_PG$finalModel$rules)) {
      SRE_train_rules_PG <- fit_rules(train_bake, drop_na(tibble(rules = RE_model_PG$finalModel$rules$description))$rules)
      SRE_test_rules_PG <- fit_rules(test_bake, drop_na(tibble(rules = RE_model_PG$finalModel$rules$description))$rules)
    } else {
      SRE_train_rules_PG <- train_bake
      SRE_test_rules_PG <- test_bake
    }
    
    
    SRE_train_PG <- cbind(SRE_train_rules_PG, fitted_smooths_train)
    SRE_test_PG <- cbind(SRE_test_rules_PG, fitted_smooths_test)
    
    ####### 
    # Only winsorize numeric features with more than 6 values
    winsorizable_PG <- get_splineworthy_columns(SRE_train_PG)
    
    #######
    # Manually create Rsample splits again, now with the splines and rules added
    indices <- list(
      list(analysis = 1:nrow(SRE_train_PG), assessment = (nrow(SRE_train_PG)+1):(nrow(SRE_train_PG)+nrow(SRE_test_PG)))
    )
    
    splits_PG <- lapply(indices, make_splits, data = rbind(SRE_train_PG, SRE_test_PG))
    
    SRE_split_PG <- manual_rset(splits_PG, c("Split SRE"))
    
    ####### 
    # Create recipes, for AUC, Brier and PG
    normalizable_PG <- colnames(training(SRE_split_PG$splits[[1]])[unlist(lapply(training(SRE_split_PG$splits[[1]]), function(x) n_distinct(x)>2))])
    
    SRE_recipe_PG <- recipe(label~., data = training(SRE_split_PG$splits[[1]])) %>%
      step_hai_winsorized_truncate(all_of(names(!!training(SRE_split_PG$splits[[1]]))[!!winsorizable_PG]), fraction = 0.025) %>%
      step_rm(all_of(names(!!training(SRE_split_PG$splits[[1]]))[!!winsorizable_PG])) %>%
      step_mutate_at(contains("winsorized"), fn = ~0.4 * ./ sd(.)) %>%
      step_mutate(across(where(is.logical), as.integer)) %>%
      step_normalize(all_of(setdiff(!!normalizable_PG, colnames(!!training(SRE_split_PG$splits[[1]])[!!winsorizable_PG])))) %>%
      step_zv()
    
    ####### 
    # Fit regular lasso for AUC, Brier, PG and extract metrics
    SRE_model_pg <- 
      parsnip::logistic_reg(
        mode = "classification",
        mixture = 1,
        penalty = best_lambda_pg
      ) %>%
      set_engine("glmnet")
    
    SRE_wf_pg <- workflow() %>%
      #add_formula(label~.) %>%
      add_recipe(SRE_recipe_PG) %>%
      add_model(SRE_model_pg)    
    
    final_SRE_fit_pg <- SRE_wf_pg %>% last_fit(SRE_split_PG$splits[[1]], metrics = metrics)
    
    pg <- final_SRE_fit_pg %>%
      collect_pg()
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "SRE", pg)
  }
  PG_results$metric<-unlist(PG_results$metric)
  write.csv(PG_results, file = paste("./results/",dataset_vector[dataset_counter],"_PG_rerun.csv", sep = ""))
  
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
