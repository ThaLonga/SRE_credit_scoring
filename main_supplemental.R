# main for De Bock homogeneous ensemble configuration and PLTR
if (!require("pacman")) install.packages("pacman") ; require("pacman")
p_load(glmnet, glmnetUtils, mgcv, MASS, tidyverse, xgboost, DiagrammeR, stringr, tictoc, doParallel, pROC, earth, Matrix, pre, caret, parsnip, ggplot2, recipes, rsample, workflows, healthyR.ai, rlang, yardstick, bonsai, lightgbm, ranger, tune, DescTools, rules, discrim, partykit)
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
    # PLTR
    #####

    full_metrics_PLTR <- list()
    full_metrics_PG_PLTR <- list()
    
    for(k in 1:nrow(inner_folds)) {
      cat("PLTR inner fold", k, "/ 5 \n")
      ####### 
      # Data is split in training and test, preprocessing is applied
      
      inner_train <- analysis(inner_folds$splits[[k]])
      inner_test <- assessment(inner_folds$splits[[k]])
      
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
        fitted_rules_inner_train <- fit_rules(inner_train_bake, rules)
        fitted_rules_inner_test <- fit_rules(inner_test_bake, rules)
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
      
      
      
      ####### 
      # Fit regular lasso for AUC, Brier, PG

      PLTR_model <- 
        parsnip::logistic_reg(
          mode = "classification",
          mixture = 1,
          penalty = tune()
        ) %>%
        set_engine("glmnet", penalty_factor = penalty_factors)
      
      PLTR_wf <- workflow() %>%
        add_recipe(PLTR_recipe) %>%
        add_model(PLTR_model)
      
      PLTR_tuned <- tune::tune_grid(
        object = PLTR_wf,
        resamples = PLTR_split,
        grid = hyperparameters_SRE_tidy, 
        metrics = metrics,
        control = tune::control_grid(verbose = TRUE, save_pred = TRUE))

      #for auc, brier
      metrics_PLTR <- PLTR_tuned$.metrics[[1]]
      metrics_PLTR$fold <- rep(k, nrow(metrics_PLTR))

      full_metrics_PLTR <- rbind(full_metrics_PLTR, metrics_PLTR)

      #for pg
      metrics_PLTR_PG <- suppressMessages(PLTR_tuned%>%collect_predictions(summarize = FALSE) %>%
                                           group_by(id, penalty, .config) %>%
                                           summarise(partial_gini = partialGini(.pred_X1, label)))
      metrics_PLTR_PG$fold <- rep(k, nrow(metrics_PLTR_PG))
      
      full_metrics_PG_PLTR <- rbind(full_metrics_PG_PLTR, metrics_PLTR_PG)
    }
    
    print("hyperparameters found")
    
    ####### 
    # Hyperparameter extraction, we use lambda.1se  = lambda.min + 1se
    aggregated_metrics <- full_metrics_PLTR %>% group_by(penalty, .config, .metric) %>%
      summarise(mean_perf = mean(.estimate))
    aggregated_metrics_PG <- full_metrics_PG_PLTR %>% group_by(penalty, .config) %>%
      summarise(mean_perf = mean(partial_gini))
    
    best_lambda_auc <- aggregated_metrics %>% filter(.metric=="roc_auc") %>% ungroup() %>% slice_max(mean_perf) %>% slice_head() %>% dplyr::select(penalty) %>% pull()
    best_lambda_brier <- aggregated_metrics %>% filter(.metric=="brier_class") %>% ungroup() %>% slice_min(mean_perf) %>% slice_head() %>% dplyr::select(penalty) %>% pull()
    best_lambda_pg <- aggregated_metrics_PG %>% ungroup() %>% slice_max(mean_perf) %>% slice_head() %>% dplyr::select(penalty) %>% pull()
    
      
    
    
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
    PLTR_train <- fit_rules(train_bake, rules)
    PLTR_test <- fit_rules(test_bake, rules)
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
    # Fit regular lasso for AUC, Brier, PG and extract metrics
    
    
    #ridge

    PLTR_model_auc <- 
      parsnip::logistic_reg(
        mode = "classification",
        mixture = 1,
        penalty = best_lambda_auc
      ) %>%
      set_engine("glmnet")
    
    PLTR_wf_auc <- workflow() %>%
      add_recipe(PLTR_recipe) %>%
      add_model(PLTR_model_auc)
    
    PLTR_model_brier <- 
      parsnip::logistic_reg(
        mode = "classification",
        mixture = 1,
        penalty = best_lambda_brier
      ) %>%
      set_engine("glmnet")
    
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
      set_engine("glmnet")
    
    PLTR_wf_pg <- workflow() %>%
      #add_formula(label~.) %>%
      add_recipe(PLTR_recipe) %>%
      add_model(PLTR_model_pg)    
    
    final_PLTR_fit_auc <- PLTR_wf_auc %>% last_fit(PLTR_split$splits[[1]], metrics = metrics)
    final_PLTR_fit_brier <- PLTR_wf_brier %>% last_fit(PLTR_split$splits[[1]], metrics = metrics)
    final_PLTR_fit_pg <- PLTR_wf_pg %>% last_fit(PLTR_split$splits[[1]], metrics = metrics)
    
    auc <- final_PLTR_fit_auc %>%
      collect_metrics() %>%
      filter(.metric == "roc_auc") %>%
      pull(.estimate)
    AUC_results[nrow(AUC_results) + 1,] = list(dataset_vector[dataset_counter], i, "PLTR", auc)
    
    brier <- final_PLTR_fit_brier %>%
      collect_metrics() %>%
      filter(.metric == "brier_class") %>%
      pull(.estimate)
    Brier_results[nrow(Brier_results) + 1,] = list(dataset_vector[dataset_counter], i, "PLTR", brier)
    
    pg <- final_PLTR_fit_pg %>%
      collect_pg()
    PG_results[nrow(PG_results) + 1,] = list(dataset_vector[dataset_counter], i, "PLTR", pg)
    
    
    
    
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
