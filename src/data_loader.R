load_data <- function () {
  for(dataset in c("german", "australian", "HMEQ", "thomas", "LC", "TC", "kaggle", "PAKDD")) {
    load(paste("data/GOLD/",dataset,".Rda", sep=""))
  }
  return(list(german, australian, HMEQ, thomas, LC, TC, kaggle, PAKDD))
}

load_results <- function (metric, dataset) {
  return(read_csv(paste("results/",dataset,"_",metric,".csv", sep=""), show_col_types = FALSE)[-1])
}

load_results <- function(directory_path = "results") {
  # List all files in the specified directory with .csv extension
  csv_files <- list.files(directory_path, pattern = "\\.csv$", full.names = TRUE)
  
  # Create an empty list to store the loaded data frames
  loaded_data <- list()
  
  # Loop through each CSV file and load it into a data frame
  for (file_path in csv_files) {
    # Extract file name without extension
    file_name <- tools::file_path_sans_ext(basename(file_path))
    
    # Load CSV file into a data frame and assign it the file name
    loaded_data[[file_name]] <- read_csv(file_path, show_col_types = FALSE)
  }
  
  return(loaded_data)
}

