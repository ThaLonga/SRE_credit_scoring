library(tidymodels, EMP)

# Define the function to calculate EMP for each fold and algorithm
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
emp_results <- calculate_emp(dataset_name, dataset_path, predictions_name, predictions_path, outerfolds, nr_repeats)
write.csv(emp_results, file = paste("./results/",dataset_vector[dataset_counter],"_RF_EMP.csv", sep = ""))
