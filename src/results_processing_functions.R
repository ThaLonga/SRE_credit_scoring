# Results processing functions

avg_ranks <- function(.data, direction = "max") {
  
  if(direction=="max") {
   .data %>%
     group_by(dataset, nr_fold) %>%
     mutate(rank = rank(-metric, ties.method = "average")) %>%
     ungroup() %>%
     group_by(dataset, algorithm) %>%
     summarise(average_rank = median(rank), average_metric = mean(metric), .groups = 'drop')
  }
  else if(direction=="min") {
    .data %>%
      group_by(dataset, nr_fold) %>%
      mutate(rank = rank(metric, ties.method = "average")) %>%
      ungroup() %>%
      group_by(dataset, algorithm) %>%
      summarise(average_rank = median(rank), average_metric = mean(metric), .groups = 'drop')
  }
  else warning("no valid direction")
}

avg_ranks_bayes <- function(.data, direction = "max") {

  if(direction=="max") {
    .data %>%
      group_by(group) %>%
      mutate(rank = rank(-metric, ties.method = "average")) %>%
      ungroup()
  }
  else if(direction=="min") {
    .data %>%
      group_by(group) %>%
      mutate(rank = rank(metric, ties.method = "average")) %>%
      ungroup()
  }
  else warning("no valid direction")
}

avg_ranks_summarized <- function(.data) {
  .data %>%
    group_by(algorithm) %>%
    summarise(average_rank = mean(average_rank))
} 

friedman_pairwise <- function(best_rank, compare_rank, N, k) {
  z = ((compare_rank - best_rank)/(sqrt(k*(k+1)/(6*N))))
  return(z)
}

format_p_values <- function(x) {
  str_replace_all(x, "\\((\\d+\\.?\\d*)\\)", function(match) {
    # Extract the number inside parentheses and format it to 3 decimals
    num <- as.numeric(str_extract(match, "\\d+\\.?\\d*"))
    formatted_num <- formatC(num, format = "f", digits = 3)
    # Remove leading zero if it exists (for numbers less than 1)
    formatted_num <- sub("^0", "", formatted_num)
    paste0("(", formatted_num, ")")
  })
}
