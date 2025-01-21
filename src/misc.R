# to import model templates to make the main code simpler 
#returns predictions

#0.01 for FM, 0.1 for others
get_splineworthy_columns <- function(X) {
  return((lapply(X, n_distinct)>20) & unlist(lapply(X, is.numeric)))
}

get_winsorizable_columns <- function(X) {
  cols <- colnames(X)[
    sapply(1:ncol(X), function(i) {
      # Check if the column is numeric and has more than two unique values
      if (is.numeric(X[, i]) && length(unique(X[, i])) > 2) {
        quantiles <- quantile(X[, i], probs = c(0.025, 1 - 0.025), na.rm = TRUE)
        quantiles[1] != quantiles[2]  # Check if quantile[1] != quantile[2]
      } else {
        FALSE  # Return FALSE for non-numeric or single-value columns
      }
    })
  ]
  
  # Exclude column names that start with "s_"
  cols <- cols[!grepl("^s\\(", cols)]
  
  return(setNames(colnames(X) %in% cols, colnames(X)))
}

#PG: cutoff = max probability of default
partialGini <- function(preds, actuals, cutoff = 0.4) {

  # Select subset with PD < 0.4
  subset_indices <- which(preds < cutoff)
  subset_preds <- preds[subset_indices]
  subset_actuals <- actuals[subset_indices]

  if(length(subset_preds)==0){
    return(0)
    print("warning: no case predictions")}
  if(length(unique(subset_actuals))<2){
    return(0.5)
    print("warning: no cases")}
  
  # Calculate ROC curve for the subset
  roc_subset <- pROC::roc(subset_actuals, subset_preds)
  
  # Calculate AUC for the subset
  partial_auc <- pROC::auc(roc_subset)
  
  # Calculate partial Gini coefficient
  partial_gini <- 2 * partial_auc - 1
  return(partial_gini)
}

delete_variables_based_on_collinearity <- function(data, target_variable, threshold = 0.7) {
  # Extract the numeric variables
  numeric_vars <- sapply(data, is.numeric)
  numeric_data <- data[, numeric_vars]
  
  # Calculate the correlation matrix
  correlation_matrix <- cor(numeric_data)
  
  # Find highly correlated variable pairs
  high_corr_pairs <- which(abs(correlation_matrix) > threshold & correlation_matrix < 1, arr.ind = TRUE)
  
  # Identify the variable with the highest correlation with the target variable in each pair
  selected_vars <- character(0)
  for (i in 1:nrow(high_corr_pairs)) {
    var1 <- rownames(correlation_matrix)[high_corr_pairs[i, 1]]
    var2 <- colnames(correlation_matrix)[high_corr_pairs[i, 2]]
    
    corr_var1 <- cor(data[[var1]], as.numeric(data[[target_variable]]))
    corr_var2 <- cor(data[[var2]], as.numeric(data[[target_variable]]))
    
    selected_var <- ifelse(abs(corr_var1) < abs(corr_var2), var1, var2)
    selected_vars <- union(selected_vars, selected_var)
  }
  return(selected_vars)
}

extractBestModel <- function(modellist, metric = "AUCROC") {
  # Extract the model with the highest mean AUCROC
  best_model <- modellist[which.max(sapply(modellist, function(x) mean(x$resample[,metric])))]
  
  # Print details of the best model
  #print(best_model)
  
  # Return the best model
  return(best_model)
}

select_best_pg_LRR <- function(.data) {
  suppressMessages({.data %>%
      collect_predictions(summarize = TRUE) %>%
      group_by(penalty, mixture, .config) %>%
      summarise(partial_gini = partialGini(.pred_X1, label)) %>%
      group_by(penalty, mixture, .config) %>%
      summarise(avg_pg = mean(partial_gini)) %>%
      ungroup() %>%
      slice_max(avg_pg) %>%
      slice_head() %>%
      dplyr::select(penalty, mixture, .config)})
}

select_best_pg_SRE <- function(.data) {
  suppressMessages({.data %>%
      collect_predictions(summarize = FALSE) %>%
      group_by(id, penalty, .config) %>%
      summarise(partial_gini = partialGini(.pred_X1, label)) %>%
      group_by(penalty, .config) %>%
      summarise(avg_pg = mean(partial_gini)) %>%
      ungroup() %>%
      slice_max(avg_pg) %>%
      slice_head() %>%
      dplyr::select(penalty, .config)})
}

select_best_pg_XGB <- function(.data) {
  suppressMessages({.data %>%
      collect_predictions(summarize = TRUE) %>%
      group_by(trees, tree_depth, learn_rate, sample_size, loss_reduction, .config) %>%
      summarise(partial_gini = partialGini(.pred_X1, label)) %>%
      group_by(trees, tree_depth, learn_rate, sample_size, loss_reduction, .config) %>%
      summarise(avg_pg = mean(partial_gini)) %>%
      ungroup() %>%
      slice_max(avg_pg) %>%
      slice_head() %>%
      dplyr::select(trees, tree_depth, learn_rate, sample_size, loss_reduction, .config)})
}

select_best_pg_LGBM <- function(.data) {
  suppressMessages({.data %>%
      collect_predictions(summarize = TRUE) %>%
      group_by(trees, tree_depth, learn_rate, sample_size, loss_reduction, .config) %>%
      summarise(partial_gini = partialGini(.pred_X1, label)) %>%
      group_by(trees, tree_depth, learn_rate, sample_size, loss_reduction, .config) %>%
      summarise(avg_pg = mean(partial_gini)) %>%
      ungroup() %>%
      slice_max(avg_pg) %>%
      slice_head() %>%
      dplyr::select(trees, tree_depth, learn_rate, sample_size, loss_reduction, .config)})
}

select_best_pg_RF <- function(.data) {
  suppressMessages({.data %>%
      collect_predictions(summarize = TRUE) %>%
      group_by(trees, mtry, min_n, .config) %>%
      summarise(partial_gini = partialGini(.pred_X1, label)) %>%
      group_by(trees, mtry, min_n, .config) %>%
      summarise(avg_pg = mean(partial_gini)) %>%
      ungroup() %>%
      slice_max(avg_pg) %>%
      slice_head() %>%
      dplyr::select(trees, mtry, min_n, .config)})
}

select_best_pg_RE <- function(.data) {
  suppressMessages({.data %>%
      collect_predictions(summarize = TRUE) %>%
      group_by(tree_depth, learn_rate, penalty, .config) %>%
      summarise(partial_gini = partialGini(.pred_X1, label)) %>%
      group_by(tree_depth, learn_rate, penalty, .config) %>%
      summarise(avg_pg = mean(partial_gini)) %>%
      ungroup() %>%
      slice_max(avg_pg) %>%
      slice_head() %>%
      dplyr::select(tree_depth, learn_rate, penalty, .config)})
}


collect_pg <- function(.data) {
  suppressMessages({.data %>%
      collect_predictions() %>%
      summarise(partial_gini = partialGini(.pred_X1, label))
  })
}

select_best_emp_LRR <- function(.data) {
  suppressMessages({.data %>%
      collect_predictions(summarize = TRUE) %>%
      group_by(penalty, mixture, .config) %>%
      summarise(emp = empCreditScoring(.pred_X1, label)$EMPC) %>%
      group_by(penalty, mixture, .config) %>%
      summarise(avg_emp = mean(emp)) %>%
      ungroup() %>%
      slice_max(avg_emp) %>%
      slice_head() %>%
      dplyr::select(penalty, mixture, .config)})
}

select_best_emp_SRE <- function(.data) {
  suppressMessages({.data %>%
      collect_predictions(summarize = FALSE) %>%
      group_by(id, penalty, .config) %>%
      summarise(emp = empCreditScoring(.pred_X1, label)$EMPC) %>%
      group_by(penalty, .config) %>%
      summarise(avg_emp = mean(emp)) %>%
      ungroup() %>%
      slice_max(avg_emp) %>%
      slice_head() %>%
      dplyr::select(penalty, .config)})
}

select_best_emp_XGB <- function(.data) {
  suppressMessages({.data %>%
      collect_predictions(summarize = TRUE) %>%
      group_by(trees, tree_depth, learn_rate, sample_size, loss_reduction, .config) %>%
      summarise(emp = empCreditScoring(.pred_X1, label)$EMPC) %>%
      group_by(trees, tree_depth, learn_rate, sample_size, loss_reduction, .config) %>%
      summarise(avg_emp = mean(emp)) %>%
      ungroup() %>%
      slice_max(avg_emp) %>%
      slice_head() %>%
      dplyr::select(trees, tree_depth, learn_rate, sample_size, loss_reduction, .config)})
}

select_best_emp_LGBM <- function(.data) {
  suppressMessages({.data %>%
      collect_predictions(summarize = TRUE) %>%
      group_by(trees, tree_depth, learn_rate, sample_size, loss_reduction, .config) %>%
      summarise(emp = empCreditScoring(.pred_X1, label)$EMPC) %>%
      group_by(trees, tree_depth, learn_rate, sample_size, loss_reduction, .config) %>%
      summarise(avg_emp = mean(emp)) %>%
      ungroup() %>%
      slice_max(avg_emp) %>%
      slice_head() %>%
      dplyr::select(trees, tree_depth, learn_rate, sample_size, loss_reduction, .config)})
}

select_best_emp_RF <- function(.data) {
  suppressMessages({.data %>%
      collect_predictions(summarize = TRUE) %>%
      group_by(trees, mtry, min_n, .config) %>%
      summarise(emp = empCreditScoring(.pred_X1, label)$EMPC) %>%
      group_by(trees, mtry, min_n, .config) %>%
      summarise(avg_emp = mean(emp)) %>%
      ungroup() %>%
      slice_max(avg_emp) %>%
      slice_head() %>%
      dplyr::select(trees, mtry, min_n, .config)})
}

select_best_emp_RE <- function(.data) {
  suppressMessages({.data %>%
      collect_predictions(summarize = TRUE) %>%
      group_by(tree_depth, learn_rate, penalty, .config) %>%
      summarise(emp = empCreditScoring(.pred_X1, label)$EMPC) %>%
      group_by(tree_depth, learn_rate, penalty, .config) %>%
      summarise(avg_emp = mean(emp)) %>%
      ungroup() %>%
      slice_max(avg_emp) %>%
      slice_head() %>%
      dplyr::select(tree_depth, learn_rate, penalty, .config)})
}


collect_emp <- function(.data) {
  suppressMessages({.data %>%
      collect_predictions() %>%
      summarise(emp = empCreditScoring(.pred_X1, label)$EMPC)
  })
}

select_best_emp_CTREE <- function(.data) {
  .data$pred %>%
    group_by(mincriterion) %>%
    summarise(emp = empCreditScoring(X1, obs)$EMPC) %>%
    group_by(mincriterion) %>%
    summarise(avg_emp = mean(emp)) %>%
    ungroup() %>%
    slice_max(avg_emp) %>%
    slice_head() %>%
    dplyr::select(mincriterion)
}

select_best_emp_RE_boosting <- function(.data) {
  .data$pred %>%
    group_by(maxdepth, learnrate) %>%
    summarise(emp = empCreditScoring(X1, obs)$EMPC) %>%
    group_by(maxdepth, learnrate) %>%
    summarise(avg_emp = mean(emp)) %>%
    ungroup() %>%
    slice_max(avg_emp) %>%
    slice_head() %>%
    dplyr::select(maxdepth, learnrate)
}

select_best_emp_RE_RF <- function(.data) {
  .data$pred %>%
    group_by(maxdepth, mtry) %>%
    summarise(emp = empCreditScoring(X1, obs)$EMPC) %>%
    group_by(maxdepth, mtry) %>%
    summarise(avg_emp = mean(emp)) %>%
    ungroup() %>%
    slice_max(avg_emp) %>%
    slice_head() %>%
    dplyr::select(maxdepth, mtry)
}

#to fit rules from pre package on dataframe that is baked with TREE_recipe
#obtain by applying "$rules$description" on model
fit_rules <- function(dataframe, rules) {
  if(!is.null(rules)) {
    # Split the rule into individual conditions
    conditions <- strsplit(rules, " & ")

    # Add 'train$' before each condition
    conditions_with_dataframe <- lapply(conditions, function(x) paste(deparse(substitute(dataframe)),"$", x, sep = ""))
    
    # Combine the conditions with ' & ' separator
    rule_list <- lapply(conditions_with_dataframe, function (x) parse(text = paste(x, collapse = " & ")))
    
    train_rules <- dataframe
    for (i in seq_along(rule_list)) {
      tryCatch({
        rule_result <- eval(rule_list[[i]])
        column_name <- paste0("rule_", i)
        train_rules[column_name] <- rule_result
      }, error = function(e) {
        warning(sprintf("Rule %d failed with error: %s", i, e$message))
        NA
      })
    }
    return(train_rules)
  } else {
    warning("No rules to fit")
  }
}



concatenate_list_of_vectors <- function(list_of_vectors) {
  # Function to concatenate a single vector with "__"
  concatenate_with_double_underscore <- function(strings) {
    concatenated_string <- paste(strings, collapse = "__")
    return(concatenated_string)
  }
  
  # Apply the concatenation function to each vector in the list
  concatenated_list <- lapply(list_of_vectors, concatenate_with_double_underscore)
  return(concatenated_list)
}

  

fit_rules_SGL <- function(dataframe, rules) {
  extract_variable_names <- function(condition) {
    # Extract all words that match column names in the dataframe
    vars <- unlist(strsplit(condition, " "))
    vars <- vars[vars %in% names(dataframe)]
    return(vars)
  }
  if(!is.null(rules)) {
    rule_vars <- concatenate_list_of_vectors(lapply(rules, extract_variable_names))
    # Split the rule into individual conditions
    conditions <- strsplit(rules, " & ")
    
    # Add 'train$' before each condition
    conditions_with_dataframe <- lapply(conditions, function(x) paste(deparse(substitute(dataframe)),"$", x, sep = ""))
    
    # Combine the conditions with ' & ' separator
    rule_list <- lapply(conditions_with_dataframe, function (x) parse(text = paste(x, collapse = " & ")))
    
    train_rules <- dataframe
    for (i in seq_along(rule_list)) {
      tryCatch({
        rule_result <- eval(rule_list[[i]])
        column_name <- paste0("rule_", i, "_", paste(rule_vars[[i]], collapse="_"))
        train_rules[column_name] <- rule_result
      }, error = function(e) {
        warning(sprintf("Rule %d failed with error: %s", i, e$message))
        NA
      })
    }
    return(train_rules)
  } else {
    warning("No rules to fit")
  }
}

fisher_score <- function(predictor, label) {
  levels <- unique(label)
  if(length(levels) != 2) stop("Fisher score is only applicable for binary classification")
  
  class1 <- label == levels[1]
  class2 <- label == levels[2]
  
  mean1 <- mean(predictor[class1], na.rm = TRUE)
  mean2 <- mean(predictor[class2], na.rm = TRUE)
  var1 <- var(predictor[class1], na.rm = TRUE)
  var2 <- var(predictor[class2], na.rm = TRUE)
  
  score <- abs(mean1 - mean2) / sqrt(var1 + var2)
  return(score)
}

fisher_score_selection <- function(data) {
  fisher_scores <- data %>%
    select(-label) %>%
    summarise(across(everything(), ~ fisher_score(.x, data$label)))
  top_n <- min(20, (ncol(data)-1))
  top_predictors <- names(sort(as.data.frame(fisher_scores)%>%unlist(), decreasing = TRUE)[1:top_n])
  return(top_predictors)
}

group_terms_by_variables <- function(terms, original_vars) {
  # Initialize a list to hold the groups
  grouped_terms <- list()
  groups <- list()
  
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
  
  return(as.numeric(groups))
}

