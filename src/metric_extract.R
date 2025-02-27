library(tidymodels, EMP)
src("./src/BigSummary.R")
src("./src/misc")

# Define the function to calculate EMP for each fold and algorithm
calculate_auc <- function(dataset_name, dataset_path, predictions_name, predictions_path, outerfolds, nr_repeats) {
  
  # Load the dataset
  load(dataset_path)
  
  # Create folds
  set.seed(111)
  
  folds <- vfold_cv(get(dataset_name), v = outerfolds, repeats = nr_repeats, strata = NULL)
  
  # Initialize an empty results data frame
  auc_results <- data.frame(
    dataset = character(),
    nr_fold = integer(),
    algorithm = character(),
    metric = numeric(),
    stringsAsFactors = FALSE
  )
  
  # Loop over each fold
  for (fold_num in seq_len(outerfolds * nr_repeats)) {
    print(fold_num)
    
    # Create the filename for the predictions CSV for the current fold
    predictions_file <- file.path(predictions_path, paste0(predictions_name, "_predictions_repeat_", fold_num, "_AUC.csv"))
    predictions <- read.csv(predictions_file)
    
    # Define a function to calculate EMP for a single algorithm
    calculate_auc_for_algorithm <- function(column_name) {
      # Calculate EMP for the current algorithm
      suppressMessages(auc <- pROC::roc(label~X1, data = data.frame(X1=predictions[[column_name]], label=assessment(folds$splits[[fold_num]])$label), direction = "<")$auc)
      return(auc)
    }
    
    # Calculate EMP in parallel for each algorithm (column in the predictions file)
    auc_metrics <- lapply(names(predictions)[-1], calculate_auc_for_algorithm)
    
    # Append results to emp_results data frame
    fold_results <- data.frame(
      dataset = predictions_name,
      nr_fold = fold_num,
      algorithm = sub("(.*)_predictions.*", "\\1", names(predictions)[-1]),
      metric = unlist(auc_metrics)
    )
    
    auc_results <- rbind(auc_results, fold_results)
  }
  
  # Return the combined results
  return(auc_results)
}

calculate_brier <- function(dataset_name, dataset_path, predictions_name, predictions_path, outerfolds, nr_repeats) {
  
  # Load the dataset
  load(dataset_path)
  
  # Create folds
  set.seed(111)
  
  folds <- vfold_cv(get(dataset_name), v = outerfolds, repeats = nr_repeats, strata = NULL)
  
  # Initialize an brierty results data frame
  brier_results <- data.frame(
    dataset = character(),
    nr_fold = integer(),
    algorithm = character(),
    metric = numeric(),
    stringsAsFactors = FALSE
  )
  
  # Loop over each fold
  for (fold_num in seq_len(outerfolds * nr_repeats)) {
    print(fold_num)
    
    # Create the filename for the predictions CSV for the current fold
    predictions_file <- file.path(predictions_path, paste0(predictions_name, "_predictions_repeat_", fold_num, "_Brier.csv"))
    predictions <- read.csv(predictions_file)
    
    # Define a function to calculate brier for a single algorithm
    calculate_brier_for_algorithm <- function(column_name) {
      # Calculate brier for the current algorithm
      brier <- brier_score(preds = predictions[[column_name]], truth = assessment(folds$splits[[fold_num]])$label)
      return(brier)
    }
    
    # Calculate brier in parallel for each algorithm (column in the predictions file)
    brier_metrics <- lapply(names(predictions)[-1], calculate_brier_for_algorithm)
    
    # Append results to brier_results data frame
    fold_results <- data.frame(
      dataset = predictions_name,
      nr_fold = fold_num,
      algorithm = sub("(.*)_predictions.*", "\\1", names(predictions)[-1]),
      metric = unlist(brier_metrics)
    )
    
    brier_results <- rbind(brier_results, fold_results)
  }
  
  # Return the combined results
  return(brier_results)
}

calculate_pg <- function(dataset_name, dataset_path, predictions_name, predictions_path, outerfolds, nr_repeats) {
  
  # Load the dataset
  load(dataset_path)
  
  # Create folds
  set.seed(111)
  
  folds <- vfold_cv(get(dataset_name), v = outerfolds, repeats = nr_repeats, strata = NULL)
  
  # Initialize an pgty results data frame
  pg_results <- data.frame(
    dataset = character(),
    nr_fold = integer(),
    algorithm = character(),
    metric = numeric(),
    stringsAsFactors = FALSE
  )
  
  # Loop over each fold
  for (fold_num in seq_len(outerfolds * nr_repeats)) {
    print(fold_num)
    
    # Create the filename for the predictions CSV for the current fold
    predictions_file <- file.path(predictions_path, paste0(predictions_name, "_predictions_repeat_", fold_num, "_PG.csv"))
    predictions <- read.csv(predictions_file)
    
    # Define a function to calculate pg for a single algorithm
    calculate_pg_for_algorithm <- function(column_name) {
      # Calculate pg for the current algorithm
      pg <- suppressMessages(partialGini(preds = predictions[[column_name]], actuals = assessment(folds$splits[[fold_num]])$label))
      return(pg)
    }
    
    # Calculate pg in parallel for each algorithm (column in the predictions file)
    pg_metrics <- lapply(names(predictions)[-1], calculate_pg_for_algorithm)
    
    # Append results to pg_results data frame
    fold_results <- data.frame(
      dataset = predictions_name,
      nr_fold = fold_num,
      algorithm = sub("(.*)_predictions.*", "\\1", names(predictions)[-1]),
      metric = unlist(pg_metrics)
    )
    
    pg_results <- rbind(pg_results, fold_results)
  }
  
  # Return the combined results
  return(pg_results)
}

calculate_emp <- function(dataset_name, dataset_path, predictions_name, predictions_path, outerfolds, nr_repeats) {
  
  # Load the dataset
  load(dataset_path)
  
  # Create folds
  set.seed(111)
  
  folds <- vfold_cv(get(dataset_name), v = outerfolds, repeats = nr_repeats, strata = NULL)
  
  # Initialize an empty results data frame
  emp_results <- data.frame(
    dataset = character(),
    nr_fold = integer(),
    algorithm = character(),
    metric = numeric(),
    stringsAsFactors = FALSE
  )
  
  # Loop over each fold
  for (fold_num in seq_len(outerfolds * nr_repeats)) {
    print(fold_num)
    
    # Create the filename for the predictions CSV for the current fold
    predictions_file <- file.path(predictions_path, paste0(predictions_name, "_predictions_repeat_", fold_num, "_EMP.csv"))
    predictions <- read.csv(predictions_file)
    
    # Define a function to calculate EMP for a single algorithm
    calculate_emp_for_algorithm <- function(column_name) {
      # Calculate EMP for the current algorithm
      emp <- EMP::empCreditScoring(predictions[[column_name]], assessment(folds$splits[[fold_num]])$label)$EMPC
      return(emp)
    }
    
    # Calculate EMP in parallel for each algorithm (column in the predictions file)
    emp_metrics <- lapply(names(predictions)[-1], calculate_emp_for_algorithm)
    
    # Append results to emp_results data frame
    fold_results <- data.frame(
      dataset = predictions_name,
      nr_fold = fold_num,
      algorithm = sub("(.*)_predictions.*", "\\1", names(predictions)[-1]),
      metric = unlist(emp_metrics)
    )
    
    emp_results <- rbind(emp_results, fold_results)
  }
  
  # Return the combined results
  return(emp_results)
}

# Usage
# Set the paths and parameters
dataset_name <- "PAKDD"
predictions_name <- "PAKDD"
dataset_path <- paste0("./data/GOLD/", dataset_name, ".Rda")  # Path to dataset Rda file
predictions_path <- "./predictions/"  # Path to the directory containing predictions files
outerfolds <- 2
nr_repeats <- 3

# Run the function
auc_results <- calculate_auc(dataset_name, dataset_path, predictions_name, predictions_path, outerfolds, nr_repeats)
brier_results <- calculate_brier(dataset_name, dataset_path, predictions_name, predictions_path, outerfolds, nr_repeats)
pg_results <- calculate_pg(dataset_name, dataset_path, predictions_name, predictions_path, outerfolds, nr_repeats)
emp_results <- calculate_emp(dataset_name, dataset_path, predictions_name, predictions_path, outerfolds, nr_repeats)
write.csv(auc_results, file = paste("./results/",dataset_name,"_v2_AUC.csv", sep = ""))
write.csv(brier_results, file = paste("./results/",dataset_name,"_v2_Brier.csv", sep = ""))
write.csv(pg_results, file = paste("./results/",dataset_name,"_v2_PG.csv", sep = ""))
write.csv(emp_results, file = paste("./results/",dataset_name,"_v2_EMP.csv", sep = ""))
