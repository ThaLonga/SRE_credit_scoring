# Functions used in preprocessing

impute_missing_by_mean <- function(data) {
  for (col in names(data)) {
    if (any(is.na(data[[col]]))) {
      col_mean <- mean(data[[col]], na.rm = TRUE)
      data[[col]][is.na(data[[col]])] <- col_mean
    }
  }
  return(data)
}

impute_missing_by_mean_with_dummy <- function(data, value = NULL) {
  for (col in names(data)) {
    if (any(is.na(data[[col]]))) {
      
      # Add dummy variable column to indicate imputed values
      imputed_col_name <- paste0(col, "_imputed")
      data[[imputed_col_name]] <- ifelse(is.na(data[[col]]), 1, 0)
      
      imputation_value <- ifelse(is.null(value), mean(data[[col]], na.rm = TRUE), value)
      data[[col]][is.na(data[[col]])] <- imputation_value
      
      print(c(imputed_col_name))
    }
  }
  return(data)
}