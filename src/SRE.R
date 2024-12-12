if (!require("pacman")) install.packages("pacman") ; require("pacman")
p_load(glmnet, glmnetUtils, mgcv, MASS, tidyverse, xgboost, DiagrammeR, stringr, tictoc, doParallel, pROC, earth, Matrix, pre, caret, parsnip, ggplot2, recipes, rsample, workflows, healthyR.ai, rlang, yardstick, bonsai, lightgbm, ranger, tune, DescTools, rules, discrim, party, partykit, sparsegl)
source("./src/hyperparameters.R")
source("./src/PLTR.R")

cv.SRE <- function(inner_folds, tree_algorithm, RE_model_AUC, RE_model_Brier, RE_model_PG, RE_model_EMP, inner_RF_list=NULL, GAM_recipe, metrics, train_bake, test_bake, regularization) {
  stopifnot(is(inner_folds, "vfold_cv")||is(inner_folds, "rset"))
  stopifnot(is(metrics, "class_prob_metric_set"))
  stopifnot(is(tree_algorithm, "character"))
  stopifnot(is(GAM_recipe, "recipe"))
  stopifnot(is(train_bake, "data.frame"))
  stopifnot(is(test_bake, "data.frame"))

  
  ####### 
  # Preprocessing on full training and test set
  # Extract fitted values for each smooth term
  smooth_terms <- grep("s\\(", unlist(str_split(as.character(formula), " \\+ ")), value = TRUE)
  smooth_terms_cleaned <- gsub("s\\(([^,]+).*", "s(\\1)", smooth_terms)
  fitted_smooths_train <- data.frame(matrix(ncol = length(smooth_terms_cleaned), nrow = nrow(train_bake)))
  fitted_smooths_test <- data.frame(matrix(ncol = length(smooth_terms_cleaned), nrow = nrow(test_bake)))
  colnames(fitted_smooths_train) <- smooth_terms_cleaned
  colnames(fitted_smooths_test) <- smooth_terms_cleaned
  cat("fitting splines\n")
  for (j in seq_along(smooth_terms_cleaned)) {
    current_smooth <- smooth_terms_cleaned[j]
    fitted_values_train <- predict(extract_fit_engine(final_GAM_fit), train_bake, type = "terms")[, current_smooth]
    fitted_smooths_train[, j] <- fitted_values_train
    fitted_values_test <- predict(extract_fit_engine(final_GAM_fit), test_bake, type = "terms")[, current_smooth]
    fitted_smooths_test[, j] <- fitted_values_test 
  }
  
  ####### 
  # Fit rules from RE_models, seperate for AUC, Brier, PG
  
  cat("fitting rules\n")
  if(identical(tree_algorithm, "PLTR")) {
    cat("training RE")
    features <- setdiff(names(train_bake), "label")  # Exclude the label column
    combinations <- combn(features, 2)  # Generate all combinations of 2 features
    rules = c()
    rules <- future.apply::future_lapply(
      1:ncol(combinations), 
      function(j) process_combination(combinations[, j], data = train_bake)
    )
    rules <- unlist(rules)
    rules <- rules[rules != ""]
   
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
  
  } else if(identical(regularization, "SGL")) {
    if(!is.null(RE_model_AUC$finalModel$rules)) {
      SRE_train_rules_AUC <- fit_rules_SGL(train_bake, drop_na(tibble(rules = RE_model_AUC$finalModel$rules$description))$rules)
      SRE_test_rules_AUC <- fit_rules_SGL(test_bake, drop_na(tibble(rules = RE_model_AUC$finalModel$rules$description))$rules)
      
      SRE_train_AUC <- cbind(SRE_train_rules_AUC, fitted_smooths_train)
      SRE_test_AUC <- cbind(SRE_test_rules_AUC, fitted_smooths_test)
    } else {
      SRE_train_AUC <- cbind(train_bake, fitted_smooths_train)
      SRE_test_AUC <- cbind(test_bake, fitted_smooths_test)
    }
    if(!is.null(RE_model_Brier$finalModel$rules)) {
      SRE_train_rules_Brier <- fit_rules_SGL(train_bake, drop_na(tibble(rules = RE_model_Brier$finalModel$rules$description))$rules)
      SRE_test_rules_Brier<- fit_rules_SGL(test_bake, drop_na(tibble(rules = RE_model_Brier$finalModel$rules$description))$rules)
      
      SRE_train_Brier <- cbind(SRE_train_rules_Brier, fitted_smooths_train)
      SRE_test_Brier <- cbind(SRE_test_rules_Brier, fitted_smooths_test)
    } else {
      SRE_train_Brier <- cbind(train_bake, fitted_smooths_train)
      SRE_test_Brier <- cbind(test_bake, fitted_smooths_test)
    }
    if(!is.null(RE_model_PG$finalModel$rules)) {
      SRE_train_rules_PG <- fit_rules_SGL(train_bake, drop_na(tibble(rules = RE_model_PG$finalModel$rules$description))$rules)
      SRE_test_rules_PG <- fit_rules_SGL(test_bake, drop_na(tibble(rules = RE_model_PG$finalModel$rules$description))$rules)
      
      SRE_train_PG <- cbind(SRE_train_rules_PG, fitted_smooths_train)
      SRE_test_PG <- cbind(SRE_test_rules_PG, fitted_smooths_test)
    } else {
      SRE_train_PG <- cbind(train_bake, fitted_smooths_train)
      SRE_test_PG <- cbind(test_bake, fitted_smooths_test)
    }
    if(!is.null(RE_model_EMP$finalModel$rules)) {
      SRE_train_rules_EMP <- fit_rules_SGL(train_bake, drop_na(tibble(rules = RE_model_EMP$finalModel$rules$description))$rules)
      SRE_test_rules_EMP <- fit_rules_SGL(test_bake, drop_na(tibble(rules = RE_model_EMP$finalModel$rules$description))$rules)
      
      SRE_train_EMP <- cbind(SRE_train_rules_EMP, fitted_smooths_train)
      SRE_test_EMP <- cbind(SRE_test_rules_EMP, fitted_smooths_test)
    } else {
      SRE_train_EMP <- cbind(train_bake, fitted_smooths_train)
      SRE_test_EMP <- cbind(test_bake, fitted_smooths_test)
    }

  } else {
    if(!is.null(RE_model_AUC$finalModel$rules)) {
      SRE_train_rules_AUC <- fit_rules(train_bake, drop_na(tibble(rules = RE_model_AUC$finalModel$rules$description))$rules)
      SRE_test_rules_AUC <- fit_rules(test_bake, drop_na(tibble(rules = RE_model_AUC$finalModel$rules$description))$rules)
      
      SRE_train_AUC <- cbind(SRE_train_rules_AUC, fitted_smooths_train)
      SRE_test_AUC <- cbind(SRE_test_rules_AUC, fitted_smooths_test)
    } else {
      SRE_train_AUC <- cbind(train_bake, fitted_smooths_train)
      SRE_test_AUC <- cbind(test_bake, fitted_smooths_test)
    }
    
    if(!is.null(RE_model_Brier$finalModel$rules)) {
      SRE_train_rules_Brier <- fit_rules(train_bake, drop_na(tibble(rules = RE_model_Brier$finalModel$rules$description))$rules)
      SRE_test_rules_Brier <- fit_rules(test_bake, drop_na(tibble(rules = RE_model_Brier$finalModel$rules$description))$rules)
      
      SRE_train_Brier <- cbind(SRE_train_rules_Brier, fitted_smooths_train)
      SRE_test_Brier <- cbind(SRE_test_rules_Brier, fitted_smooths_test)
      
    } else {
      SRE_train_Brier <- cbind(train_bake, fitted_smooths_train)
      SRE_test_Brier <- cbind(test_bake, fitted_smooths_test)
    }
    
    if(!is.null(RE_model_PG$finalModel$rules)) {
      SRE_train_rules_PG <- fit_rules(train_bake, drop_na(tibble(rules = RE_model_PG$finalModel$rules$description))$rules)
      SRE_test_rules_PG <- fit_rules(test_bake, drop_na(tibble(rules = RE_model_PG$finalModel$rules$description))$rules)
      
      SRE_train_PG <- cbind(SRE_train_rules_PG, fitted_smooths_train)
      SRE_test_PG <- cbind(SRE_test_rules_PG, fitted_smooths_test)
    } else {
      SRE_train_PG <- cbind(train_bake, fitted_smooths_train)
      SRE_test_PG <- cbind(test_bake, fitted_smooths_test)
    }
    
    if(!is.null(RE_model_EMP$finalModel$rules)) {
      SRE_train_rules_EMP <- fit_rules(train_bake, drop_na(tibble(rules = RE_model_EMP$finalModel$rules$description))$rules)
      SRE_test_rules_EMP <- fit_rules(test_bake, drop_na(tibble(rules = RE_model_EMP$finalModel$rules$description))$rules)
      
      SRE_train_EMP <- cbind(SRE_train_rules_EMP, fitted_smooths_train)
      SRE_test_EMP <- cbind(SRE_test_rules_EMP, fitted_smooths_test)
      
    } else {
      SRE_train_EMP <- cbind(train_bake, fitted_smooths_train)
      SRE_test_EMP <- cbind(test_bake, fitted_smooths_test)
    }
  }
  
  
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
  
  



  if(identical(regularization, "SGL")) {
    AUC_SGL_table_train <- SRE_recipe_AUC %>%prep()%>% bake(training(SRE_split_AUC$splits[[1]]))
    Brier_SGL_table_train <- SRE_recipe_Brier %>%prep()%>% bake(training(SRE_split_Brier$splits[[1]]))
    PG_SGL_table_train <- SRE_recipe_PG %>%prep()%>% bake(training(SRE_split_PG$splits[[1]]))
    EMP_SGL_table_train <- SRE_recipe_EMP %>%prep()%>% bake(training(SRE_split_EMP$splits[[1]]))      
    
    AUC_SGL_table_test <- SRE_recipe_AUC %>%prep()%>% bake(assessment(SRE_split_AUC$splits[[1]]))
    Brier_SGL_table_test <- SRE_recipe_Brier %>%prep()%>% bake(assessment(SRE_split_Brier$splits[[1]]))
    PG_SGL_table_test <- SRE_recipe_PG %>%prep()%>% bake(assessment(SRE_split_PG$splits[[1]]))
    EMP_SGL_table_test <- SRE_recipe_EMP %>%prep()%>% bake(assessment(SRE_split_EMP$splits[[1]]))
    
    AUC_SGL_table_train_x <- AUC_SGL_table_train %>% select(-label)
    Brier_SGL_table_train_x <- Brier_SGL_table_train %>% select(-label)
    PG_SGL_table_train_x <- PG_SGL_table_train %>% select(-label)
    EMP_SGL_table_train_x <- EMP_SGL_table_train %>% select(-label)
    
    AUC_groups <- group_terms_by_variables(names(AUC_SGL_table_train_x), names(train))
    Brier_groups <- group_terms_by_variables(names(Brier_SGL_table_train_x), names(train))
    PG_groups <- group_terms_by_variables(names(PG_SGL_table_train_x), names(train))
    EMP_groups <- group_terms_by_variables(names(EMP_SGL_table_train_x), names(train))
    
    
    
    data_list <- list(list(AUC_SGL_table_train_x, AUC_groups, AUC_SGL_table_train, best_lambda_auc),
                      list(Brier_SGL_table_train_x, Brier_groups, Brier_SGL_table_train, best_lambda_brier),
                      list(PG_SGL_table_train_x, PG_groups, PG_SGL_table_train, best_lambda_pg),
                      list(EMP_SGL_table_train_x, EMP_groups, EMP_SGL_table_train, best_lambda_emp))
    
    clusterExport(cl, list("sparsegl"))
    
    fit_sparsegl_parallel <- function(data_subset) {
      # Unpack the components of the data_subset (this is a list of 4 elements)
      train_x <- data_subset[[1]]      # Training data (X)
      groups <- data_subset[[2]]       # Groups (lambda)
      train <- data_subset[[3]]        # Training labels
      lambda <- data_subset[[4]]

      # Fit the sparsegl model for the current dataset subset
      glm_model <- sparsegl(
        as.matrix(train_x)[, order(groups)],  # Reorder according to 'groups'
        train$label,                          # Assuming 'train' has a 'label' column
        #sort(groups),                         # Sorting groups
        seq(1:length(groups)),
        family = "binomial",                  # Binary logistic regression
        lambda = lambda,
        asparse = 0.5
      )
      
      return(glm_model)
    }
    
    # Run the evaluation in parallel across all lambda values
    tic()
    results <- parLapply(cl, data_list, fit_sparsegl_parallel)
    toc()

    AUC_lambdas <- data.frame(matrix(nrow = 0, ncol = 2))
    Brier_lambdas <- data.frame(matrix(nrow = 0, ncol = 2))
    PG_lambdas <- data.frame(matrix(nrow = 0, ncol = 2))
    EMP_lambdas <- data.frame(matrix(nrow = 0, ncol = 2))

    for(s in results[[1]]$lambda) {
      SRE_SGL_preds <- predict(results[[1]], s = s, newx = as.matrix(dplyr::select(AUC_SGL_table_test, -label)), type = "response")
      AUC_data <- data.frame(x1=SRE_SGL_preds, label=AUC_SGL_table_test$label)
      g <- roc(label ~ s1, data = AUC_data, direction = "<")
      AUC <- g$auc
      AUC_lambdas[nrow(AUC_lambdas) + 1,] = c(s, AUC)
    }
    for(s in results[[2]]$lambda) {
      SRE_SGL_preds <- predict(results[[2]], s = s, newx = as.matrix(dplyr::select(Brier_SGL_table_test, -label)), type = "response")
      Brier_data <- data.frame(x1=SRE_SGL_preds, label=Brier_SGL_table_test$label)
      Brier <- brier_class_vec(truth = Brier_data$label, estimate = Brier_data$s1)
      Brier_lambdas[nrow(Brier_lambdas) + 1,] = c(s, Brier)
    }
    for(s in results[[3]]$lambda) {
      SRE_SGL_preds <- predict(results[[3]], s = s, newx = as.matrix(dplyr::select(PG_SGL_table_test, -label)), type = "response")
      PG_data <- data.frame(x1=SRE_SGL_preds, label=PG_SGL_table_test$label)
      PG <- partialGini(PG_data$x1, PG_data$label)
      PG_lambdas[nrow(PG_lambdas) + 1,] = c(s, PG)
    }
    for(s in results[[4]]$lambda) {
      SRE_SGL_preds <- predict(results[[4]], s = s, newx = as.matrix(dplyr::select(EMP_SGL_table_test, -label)), type = "response")
      EMP_data <- data.frame(x1=SRE_SGL_preds, label=EMP_SGL_table_test$label)
      EMP <- partialGini(EMP_data$x1, EMP_data$label)
      EMP_lambdas[nrow(EMP_lambdas) + 1,] = c(s, EMP)
    }
  } else {
    
    #####      #####      #####      #####      #####      #####      #####      #####      #####      #####      #####      #####      #####      #####      #####      #####      #####      #####      #####      #####      #####
    # cv.glmnet
    #####
    
    AUC_SGL_table_train <- SRE_recipe_AUC %>%prep()%>% bake(training(SRE_split_AUC$splits[[1]]))
    Brier_SGL_table_train <- SRE_recipe_Brier %>%prep()%>% bake(training(SRE_split_Brier$splits[[1]]))
    PG_SGL_table_train <- SRE_recipe_PG %>%prep()%>% bake(training(SRE_split_PG$splits[[1]]))
    EMP_SGL_table_train <- SRE_recipe_EMP %>%prep()%>% bake(training(SRE_split_EMP$splits[[1]]))      
    
    AUC_SGL_table_test <- SRE_recipe_AUC %>%prep()%>% bake(assessment(SRE_split_AUC$splits[[1]]))
    Brier_SGL_table_test <- SRE_recipe_Brier %>%prep()%>% bake(assessment(SRE_split_Brier$splits[[1]]))
    PG_SGL_table_test <- SRE_recipe_PG %>%prep()%>% bake(assessment(SRE_split_PG$splits[[1]]))
    EMP_SGL_table_test <- SRE_recipe_EMP %>%prep()%>% bake(assessment(SRE_split_EMP$splits[[1]]))
    
    

    cv.glmnet.args <- list()

    cv.glmnet.args$weights <- NULL
    cv.glmnet.args$family <- "binomial"
    cv.glmnet.args$parallel <- TRUE
    cv.glmnet.args$standardize <- FALSE
    cv.glmnet.args$alpha <- 0
    cv.glmnet.args$relax <- FALSE
    cv.glmnet.args$foldid <- sample(rep(1:3, length = nrow(train_bake))) 
    
    cv.glmnet.args_AUC <- cv.glmnet.args_Brier <- cv.glmnet.args_PG <- cv.glmnet.args_EMP <- cv.glmnet.args
    
    cv.glmnet.args_AUC$x <- as.matrix(AUC_SGL_table_train %>% select(-label))
    cv.glmnet.args_AUC$y <- AUC_SGL_table_train$label
    cv.glmnet.args_AUC$penalty.factor <- rep(1L, times = ncol(AUC_SGL_table_train %>% select(-label)))
    cv.glmnet.args_AUC$type.measure <- "auc"
    
    cv.glmnet.args_Brier$x <- as.matrix(Brier_SGL_table_train %>% select(-label))
    cv.glmnet.args_Brier$y <- Brier_SGL_table_train$label
    cv.glmnet.args_Brier$penalty.factor <- rep(1L, times = ncol(Brier_SGL_table_train %>% select(-label)))
    cv.glmnet.args_AUC$type.measure <- "mse"
    
    cv.glmnet.args_PG$x <- as.matrix(PG_SGL_table_train %>% select(-label))
    cv.glmnet.args_PG$y <- PG_SGL_table_train$label
    cv.glmnet.args_PG$penalty.factor <- rep(1L, times = ncol(PG_SGL_table_train %>% select(-label)))
    cv.glmnet.args_AUC$type.measure <- "auc"
    
    cv.glmnet.args_EMP$x <- as.matrix(EMP_SGL_table_train %>% select(-label))
    cv.glmnet.args_EMP$y <- EMP_SGL_table_train$label
    cv.glmnet.args_EMP$penalty.factor <- rep(1L, times = ncol(EMP_SGL_table_train %>% select(-label)))
    
    ## Compute penalty weights
    glmnet.fit_AUC <- do.call(cv.glmnet, cv.glmnet.args_AUC)
    glmnet.fit_Brier <- do.call(cv.glmnet, cv.glmnet.args_Brier)
    glmnet.fit_PG <- do.call(cv.glmnet, cv.glmnet.args_PG)
    glmnet.fit_EMP <- do.call(cv.glmnet, cv.glmnet.args_EMP)
    
    penalty.factor_AUC <- coef(glmnet.fit_AUC)
    penalty.factor_Brier <- coef(glmnet.fit_Brier)
    penalty.factor_PG <- coef(glmnet.fit_PG)
    penalty.factor_EMP <- coef(glmnet.fit_EMP)

    if (is.list(penalty.factor_AUC)) {
      penalty.factor_AUC <- rowMeans(do.call(cbind, penalty.factor_AUC))
      penalty.factor_AUC <- 1L / abs(as.numeric(penalty.factor_AUC)[
        -which(names(penalty.factor_AUC) == "(Intercept)")])
    } else {
      penalty.factor_AUC <- 1L / abs(as.numeric(penalty.factor_AUC)[
        -which(rownames(penalty.factor_AUC) == "(Intercept)")])
    }
    if (is.list(penalty.factor_Brier)) {
      penalty.factor_Brier <- rowMeans(do.call(cbind, penalty.factor_Brier))
      penalty.factor_Brier <- 1L / abs(as.numeric(penalty.factor_Brier)[
        -which(names(penalty.factor_Brier) == "(Intercept)")])
    } else {
      penalty.factor_Brier <- 1L / abs(as.numeric(penalty.factor_Brier)[
        -which(rownames(penalty.factor_Brier) == "(Intercept)")])
    }
    if (is.list(penalty.factor_PG)) {
      penalty.factor_PG <- rowMeans(do.call(cbind, penalty.factor_PG))
      penalty.factor_PG <- 1L / abs(as.numeric(penalty.factor_PG)[
        -which(names(penalty.factor_PG) == "(Intercept)")])
    } else {
      penalty.factor_PG <- 1L / abs(as.numeric(penalty.factor_PG)[
        -which(rownames(penalty.factor_PG) == "(Intercept)")])
    }
    if (is.list(penalty.factor_EMP)) {
      penalty.factor_EMP <- rowMeans(do.call(cbind, penalty.factor_EMP))
      penalty.factor_EMP <- 1L / abs(as.numeric(penalty.factor_EMP)[
        -which(names(penalty.factor_EMP) == "(Intercept)")])
    } else {
      penalty.factor_EMP <- 1L / abs(as.numeric(penalty.factor_EMP)[
        -which(rownames(penalty.factor_EMP) == "(Intercept)")])
    }
    
    ## Fit final model
    cv.glmnet.args$alpha <- 1
    cv.glmnet.args$relax <- FALSE
    cv.glmnet.args$foldid <- sample(rep(1:3, length = nrow(train_bake))) 
    cv.glmnet.args_AUC <- cv.glmnet.args_Brier <- cv.glmnet.args_PG <- cv.glmnet.args_EMP <- cv.glmnet.args
    
    cv.glmnet.args_AUC$x <- as.matrix(AUC_SGL_table_train %>% select(-label))
    cv.glmnet.args_AUC$y <- AUC_SGL_table_train$label
    cv.glmnet.args_Brier$x <- as.matrix(Brier_SGL_table_train %>% select(-label))
    cv.glmnet.args_Brier$y <- Brier_SGL_table_train$label
    cv.glmnet.args_PG$x <- as.matrix(PG_SGL_table_train %>% select(-label))
    cv.glmnet.args_PG$y <- PG_SGL_table_train$label
    cv.glmnet.args_EMP$x <- as.matrix(EMP_SGL_table_train %>% select(-label))
    cv.glmnet.args_EMP$y <- EMP_SGL_table_train$label
    
    cv.glmnet.args_AUC$penalty.factor <- penalty.factor_AUC
    cv.glmnet.args_Brier$penalty.factor <- penalty.factor_Brier
    cv.glmnet.args_PG$penalty.factor <- penalty.factor_PG
    cv.glmnet.args_EMP$penalty.factor <- penalty.factor_EMP
    
    cv.glmnet.args_AUC$type.measure <- "auc"
    cv.glmnet.args_Brier$type.measure <- "mse"
    cv.glmnet.args_PG$type.measure <- "auc"
    

    glmnet.fit_adalasso_AUC <- do.call(cv.glmnet, cv.glmnet.args_AUC)
    glmnet.fit_adalasso_Brier <- do.call(cv.glmnet, cv.glmnet.args_Brier)
    glmnet.fit_adalasso_PG <- do.call(cv.glmnet, cv.glmnet.args_PG)
    glmnet.fit_adalasso_EMP <- do.call(cv.glmnet, cv.glmnet.args_EMP)
    
    AUC_preds <- predict(glmnet.fit_adalasso_AUC, newx=as.matrix(AUC_SGL_table_test%>%select(-label)), s="lambda.min", type="response")
    Brier_preds <- predict(glmnet.fit_adalasso_Brier, newx=as.matrix(Brier_SGL_table_test%>%select(-label)), s="lambda.min", type="response")
    PG_preds <- predict(glmnet.fit_adalasso_PG, newx=as.matrix(PG_SGL_table_test%>%select(-label)), s="lambda.min", type="response")
    EMP_preds <- predict(glmnet.fit_adalasso_EMP, newx=as.matrix(EMP_SGL_table_test%>%select(-label)), s="lambda.min", type="response")
    
  }

    #####      #####      #####      #####      #####      #####      #####      #####      #####      #####      #####      #####      #####      #####      #####      #####      #####      #####      #####      #####      #####
  
  return(list(best_AUC = AUC_preds, best_Brier = Brier_preds, best_PG = PG_preds, best_EMP = EMP_preds))
  }
