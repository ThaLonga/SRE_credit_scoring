# main for De Bock homogeneous ensemble configuration and PLTR
if (!require("pacman")) install.packages("pacman") ; require("pacman")
p_load(glmnet, glmnetUtils, mgcv, MASS, tidyverse, xgboost, DiagrammeR, stringr, tictoc, doParallel, pROC, earth, Matrix, pre, caret, parsnip, ggplot2, recipes, rsample, workflows, healthyR.ai, rlang, yardstick, bonsai, lightgbm, ranger, tune, DescTools, rules, discrim, partykit)
#conventions:
# target variable = label
# indepentent variables = all others

#levels should be X1 and X2
#evaluation function for caret tuning from pre package
source("./src/misc.R")
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

for(dataset in datasets[1]) {
  
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
    set.seed(innerseed)
    inner_split <- train %>% validation_split(prop=3/4)
    inner_ids_in <- inner_split$splits[[1]]$in_id
    
    
    ############################################################################
    # SINGLE CLASSIFIERS
    ############################################################################
    
    
    
    #####
    # PLTR
    #####
    
    cv.PLTR <- function(inner_split, metrics, train_bake, test_bake) {
      stopifnot(is(inner_split, "vfold_cv")||is(inner_split, "rset"))
      stopifnot(is(metrics, "class_prob_metric_set"))
      stopifnot(is(GAM_recipe, "recipe"))
      stopifnot(is(train_bake, "data.frame"))
      stopifnot(is(test_bake, "data.frame"))
      
      
      full_metrics_PLTR_ridge <- list()
      full_metrics_PG_PLTR_ridge <- list()
      
      full_metrics_AUC <- list()
      full_metrics_Brier <- list()
      full_metrics_PG <- list()
      
      
      
      for(k in 1:nrow(inner_split)) {
        cat("PLTR inner fold", k, "/ ", nrow(inner_split), "\n")
        ####### 
        # Data is split in training and test, preprocessing is applied
        
        inner_train <- analysis(inner_split$splits[[k]])
        inner_test <- assessment(inner_split$splits[[k]])
        
        inner_train_bake <- XGB_recipe %>% prep(inner_train) %>% bake(inner_train)
        inner_test_bake <- XGB_recipe %>% prep(inner_train) %>% bake(inner_test)
        
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
        if(!is_empty(rules)) {
          fitted_rules_inner_train <- fit_rules(inner_train_bake, unique(rules))
          fitted_rules_inner_test <- fit_rules(inner_test_bake, unique(rules))
        }
        indices <- list(
          list(analysis = 1:nrow(fitted_rules_inner_train), assessment = (nrow(fitted_rules_inner_train)+1):(nrow(fitted_rules_inner_train)+nrow(fitted_rules_inner_test)))
        )
        
        splits <- lapply(indices, make_splits, data = rbind(fitted_rules_inner_train, fitted_rules_inner_test))
        PLTR_split <- manual_rset(splits, c("Split_PLTR"))
        
        
        PLTR_recipe <- recipe(label~., data = training(PLTR_split$splits[[1]])) %>%
          step_mutate(across(where(is.logical), as.integer)) %>%
          step_corr(all_predictors(), threshold=0.9) %>%
          step_zv()
        
        ######
        # Fit ridge for adalasso
        
        ridge_model <- 
          parsnip::logistic_reg(
            mode = "classification",
            mixture = 0,
            penalty = tune()
          ) %>%
          set_engine("glmnet")
        
        PLTR_ridge_wf <- workflow() %>%
          add_recipe(PLTR_recipe) %>%
          add_model(ridge_model)
        
        
        PLTR_ridge_tuned <- tune::tune_grid(
          object = PLTR_ridge_wf,
          resamples = PLTR_split,
          grid = hyperparameters_SRE_tidy, 
          metrics = metrics,
          control = tune::control_grid(verbose = TRUE, save_pred = TRUE))
        
        #for auc, brier
        metrics_PLTR_ridge <- PLTR_ridge_tuned$.metrics[[1]]
        metrics_PLTR_ridge$fold <- rep(k, nrow(PLTR_ridge_tuned))
        
        full_metrics_PLTR_ridge <- rbind(full_metrics_PLTR_ridge, metrics_PLTR_ridge)
        
        #for pg
        metrics_PLTR_ridge_PG <- suppressMessages(PLTR_ridge_tuned%>%collect_predictions(summarize = FALSE) %>%
                                                    group_by(id, penalty, .config) %>%
                                                    summarise(partial_gini = partialGini(.pred_X1, label)))
        metrics_PLTR_ridge_PG$fold <- rep(k, nrow(metrics_PLTR_ridge_PG))
        
        full_metrics_PG_PLTR_ridge <- rbind(full_metrics_PG_PLTR_ridge, metrics_PLTR_ridge_PG)
        
        best_ridge_auc <- PLTR_ridge_tuned %>% select_best(metric="roc_auc")
        final_ridge_wf_auc <- PLTR_ridge_wf %>% finalize_workflow(best_ridge_auc)
        final_ridge_fit_auc <- final_ridge_wf_auc %>% last_fit(PLTR_split$splits[[k]], metrics = metrics)
        ridge_penalties_AUC <- coef((final_ridge_fit_auc%>%extract_fit_engine()), s=best_ridge_auc$penalty)
        
        best_ridge_Brier <- PLTR_ridge_tuned %>% select_best(metric="brier_class")
        final_ridge_wf_Brier <- PLTR_ridge_wf %>% finalize_workflow(best_ridge_Brier)
        final_ridge_fit_Brier <- final_ridge_wf_Brier %>% last_fit(PLTR_split$splits[[k]], metrics = metrics)
        ridge_penalties_Brier <- coef((final_ridge_fit_Brier%>%extract_fit_engine()), s=best_ridge_Brier$penalty)
        
        best_ridge_PG <- PLTR_ridge_tuned %>% select_best_pg_SRE()
        final_ridge_wf_PG <- PLTR_ridge_wf %>% finalize_workflow(best_ridge_PG)
        final_ridge_fit_PG <- final_ridge_wf_PG %>% last_fit(PLTR_split$splits[[k]], metrics = metrics)
        ridge_penalties_PG <- coef((final_ridge_fit_PG%>%extract_fit_engine()), s=best_ridge_PG$penalty)
        
        ####### 
        # Fit regular lasso for AUC, Brier, PG
        
        PLTR_model_AUC <- 
          parsnip::logistic_reg(
            mode = "classification",
            mixture = 1,
            penalty = tune()
          ) %>%
          set_engine("glmnet", penalty_factor = 1/abs(ridge_penalties_AUC[-1]))
        
        PLTR_model_Brier <- 
          parsnip::logistic_reg(
            mode = "classification",
            mixture = 1,
            penalty = tune()
          ) %>%
          set_engine("glmnet", penalty_factor = 1/abs(ridge_penalties_Brier[-1]))
        
        PLTR_model_PG <- 
          parsnip::logistic_reg(
            mode = "classification",
            mixture = 1,
            penalty = tune()
          ) %>%
          set_engine("glmnet", penalty_factor = 1/abs(ridge_penalties_PG[-1]))
        
        PLTR_wf_AUC <- workflow() %>%
          add_recipe(PLTR_recipe) %>%
          add_model(PLTR_model_AUC)
        
        PLTR_wf_Brier <- workflow() %>%
          add_recipe(PLTR_recipe) %>%
          add_model(PLTR_model_Brier)
        
        PLTR_wf_PG <- workflow() %>%
          add_recipe(PLTR_recipe) %>%
          add_model(PLTR_model_PG)
        
        PLTR_tuned_AUC <- tune::tune_grid(
          object = PLTR_wf_AUC,
          resamples = PLTR_split,
          grid = hyperparameters_SRE_tidy, 
          metrics = metrics,
          control = tune::control_grid(verbose = TRUE, save_pred = TRUE))
        
        PLTR_tuned_Brier <- tune::tune_grid(
          object = PLTR_wf_Brier,
          resamples = PLTR_split,
          grid = hyperparameters_SRE_tidy, 
          metrics = metrics,
          control = tune::control_grid(verbose = TRUE, save_pred = TRUE))
        
        PLTR_tuned_PG <- tune::tune_grid(
          object = PLTR_wf_PG,
          resamples = PLTR_split,
          grid = hyperparameters_SRE_tidy, 
          metrics = metrics,
          control = tune::control_grid(verbose = TRUE, save_pred = TRUE))
        
        #for auc, brier
        metrics_PLTR_AUC <- PLTR_tuned_AUC$.metrics[[1]]
        metrics_PLTR_Brier <- PLTR_tuned_Brier$.metrics[[1]]
        metrics_PLTR_AUC$fold <- rep(k, nrow(metrics_PLTR_AUC))
        metrics_PLTR_Brier$fold <- rep(k, nrow(metrics_PLTR_Brier))
        
        full_metrics_AUC <- rbind(full_metrics_AUC, metrics_PLTR_AUC)
        full_metrics_Brier <- rbind(full_metrics_Brier, metrics_PLTR_Brier)
        
        #for pg
        metrics_PLTR_PG <- suppressMessages(PLTR_tuned_PG%>%collect_predictions(summarize = FALSE) %>%
                                              group_by(id, penalty, .config) %>%
                                              summarise(partial_gini = partialGini(.pred_X1, label)))
        metrics_PLTR_PG$fold <- rep(k, nrow(metrics_PLTR_PG))
        
        full_metrics_PG <- rbind(full_metrics_PG, metrics_PLTR_PG)
      }
      
      print("hyperparameters found")
      
      ####### 
      # Hyperparameter extraction, we use lambda.1se  = lambda.min + 1se
      # Ridge
      aggregated_metrics_ridge_AUC_Brier <- full_metrics_PLTR_ridge %>% group_by(penalty, .config, .metric) %>%
        summarise(mean_perf = mean(.estimate))
      aggregated_metrics_ridge_PG <- full_metrics_PG_PLTR_ridge %>% group_by(penalty, .config) %>%
        summarise(mean_perf = mean(partial_gini))
      
      best_lambda_ridge_auc <- aggregated_metrics_ridge_AUC_Brier %>% filter(.metric=="roc_auc") %>% ungroup() %>% slice_max(mean_perf) %>% slice_head() %>% dplyr::select(penalty) %>% pull()
      best_lambda_ridge_brier <- aggregated_metrics_ridge_AUC_Brier %>% filter(.metric=="brier_class") %>% ungroup() %>% slice_min(mean_perf) %>% slice_head() %>% dplyr::select(penalty) %>% pull()
      best_lambda_ridge_pg <- aggregated_metrics_ridge_PG %>% ungroup() %>% slice_max(mean_perf) %>% slice_head() %>% dplyr::select(penalty) %>% pull()
      
      
      # Lasso
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
        PLTR_train <- fit_rules(train_bake, unique(rules))
        PLTR_test <- fit_rules(test_bake, unique(rules))
      }
      
      #######
      # Manually create Rsample splits again, now with the splines and rules added
      indices <- list(
        list(analysis = 1:nrow(PLTR_train), assessment = (nrow(PLTR_train)+1):(nrow(PLTR_train)+nrow(PLTR_test)))
      )
      
      splits_PLTR <- lapply(indices, make_splits, data = rbind(PLTR_train, PLTR_test))
      
      PLTR_split <- manual_rset(splits_PLTR, c("Split SRE"))
      
      
      PLTR_recipe <- recipe(label~., data = training(PLTR_split$splits[[1]])) %>%
        step_mutate(across(where(is.logical), as.integer)) %>%
        step_corr(all_predictors(), threshold = 0.99) %>%
        step_zv()
      
      
      #######
      # Ridge for alasso
      
      PLTR_model_ridge_auc <- 
        parsnip::logistic_reg(
          mode = "classification",
          mixture = 0,
          penalty = best_lambda_ridge_auc
        ) %>%
        set_engine("glmnet")
      
      PLTR_wf_ridge_auc <- workflow() %>%
        #add_formula(label~.) %>%
        add_recipe(PLTR_recipe) %>%
        add_model(PLTR_model_ridge_auc)
      
      PLTR_model_ridge_brier <- 
        parsnip::logistic_reg(
          mode = "classification",
          mixture = 0,
          penalty = best_lambda_ridge_brier
        ) %>%
        set_engine("glmnet")
      
      PLTR_wf_ridge_brier <- workflow() %>%
        #add_formula(label~.) %>%
        add_recipe(PLTR_recipe) %>%
        add_model(PLTR_model_ridge_brier)    
      
      PLTR_model_ridge_pg <- 
        parsnip::logistic_reg(
          mode = "classification",
          mixture = 0,
          penalty = best_lambda_ridge_pg    #sd can be very high for PG resulting in way too high lambda, leaving only the intercept
        ) %>%
        set_engine("glmnet")
      
      PLTR_wf_ridge_pg <- workflow() %>%
        #add_formula(label~.) %>%
        add_recipe(PLTR_recipe) %>%
        add_model(PLTR_model_ridge_pg)    
      
      final_ridge_fit_auc <- PLTR_wf_ridge_auc %>% last_fit(PLTR_split$splits[[1]], metrics = metrics)
      final_ridge_fit_brier <- PLTR_wf_ridge_brier %>% last_fit(PLTR_split$splits[[1]], metrics = metrics)
      final_ridge_fit_pg <- PLTR_wf_ridge_pg %>% last_fit(PLTR_split$splits[[1]], metrics = metrics)
      
      ridge_penalties_AUC <- coef((final_ridge_fit_auc%>%extract_fit_engine()), s=best_lambda_ridge_auc)
      ridge_penalties_Brier <- coef((final_ridge_fit_Brier%>%extract_fit_engine()), s=best_lambda_ridge_brier)
      ridge_penalties_PG <- coef((final_ridge_fit_PG%>%extract_fit_engine()), s=best_lambda_ridge_pg)
      
      
      
      
      
      ####### 
      # Fit regular lasso for AUC, Brier, PG and extract metrics
      
      
      PLTR_model_auc <- 
        parsnip::logistic_reg(
          mode = "classification",
          mixture = 1,
          penalty = best_lambda_auc
        ) %>%
        set_engine("glmnet", penalty.factors = 1 / abs(ridge_penalties_AUC[-1]))
      
      PLTR_wf_auc <- workflow() %>%
        add_recipe(PLTR_recipe) %>%
        add_model(PLTR_model_auc)
      
      PLTR_model_brier <- 
        parsnip::logistic_reg(
          mode = "classification",
          mixture = 1,
          penalty = best_lambda_brier
        ) %>%
        set_engine("glmnet", penalty.factors = 1 / abs(ridge_penalties_Brier[-1]))
      
      PLTR_wf_brier <- workflow() %>%
        #add_formula(label~.) %>%
        add_recipe(PLTR_recipe) %>%
        add_model(PLTR_model_brier)    
      
      PLTR_model_pg <- 
        parsnip::logistic_reg(
          mode = "classification",
          mixture = 1,
          penalty = best_lambda_pg    #sd can be very high for PG resulting in way too high lambda, leaving only the intercept
        ) %>%
        set_engine("glmnet", penalty.factors = 1 / abs(ridge_penalties_PG[-1]))
      
      PLTR_wf_pg <- workflow() %>%
        #add_formula(label~.) %>%
        add_recipe(PLTR_recipe) %>%
        add_model(PLTR_model_pg)    
      
      final_PLTR_fit_auc <- PLTR_wf_auc %>% last_fit(PLTR_split$splits[[1]], metrics = metrics)
      final_PLTR_fit_brier <- PLTR_wf_brier %>% last_fit(PLTR_split$splits[[1]], metrics = metrics)
      final_PLTR_fit_pg <- PLTR_wf_pg %>% last_fit(PLTR_split$splits[[1]], metrics = metrics)
      return(list(best_AUC = final_PLTR_fit_auc, best_Brier = final_PLTR_fit_brier, best_PG = final_PLTR_fit_pg))
    }
    
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
    
    # Predictions
    PLTR_predictions_AUC <- PLTR$best_AUC$.predictions[[1]]$.pred_X1
    PLTR_predictions_Brier <- PLTR$best_Brier$.predictions[[1]]$.pred_X1
    PLTR_predictions_PG <- PLTR$best_PG$.predictions[[1]]$.pred_X1
    
    
    write.csv(PLTR_predictions_AUC, file = paste("./predictions/",dataset_vector[dataset_counter],"_predictions_PLTR_repeat_", i, "_AUC.csv", sep = ""))
    write.csv(PLTR_predictions_Brier, file = paste("./predictions/",dataset_vector[dataset_counter],"_predictions_PLTR_repeat_", i, "_Brier.csv", sep = ""))
    write.csv(PLTR_predictions_PG, file = paste("./predictions/",dataset_vector[dataset_counter],"_predictions_PLTR_repeat_", i, "_PG.csv", sep = ""))
    
    
  }
  write.csv(AUC_results, file = paste("./results_supp/",dataset_vector[dataset_counter],"_AUC_PLTR.csv", sep = ""))
  write.csv(Brier_results, file = paste("./results_supp/",dataset_vector[dataset_counter],"_BRIER_PLTR.csv", sep = ""))
  PG_results$metric<-unlist(PG_results$metric)
  write.csv(PG_results, file = paste("./results_supp/",dataset_vector[dataset_counter],"_PG_PLTR.csv", sep = ""))
  
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
