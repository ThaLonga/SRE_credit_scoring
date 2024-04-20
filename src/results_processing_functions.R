# Results processing functions

avg_ranks <- function(.data, direction = "max") {
  
  if(direction=="max") {
   .data %>%
     group_by(dataset, nr_fold) %>%
     mutate(rank = rank(-metric, ties.method = "average")) %>%
     ungroup() %>%
     group_by(dataset, algorithm) %>%
     summarise(average_rank = mean(rank), average_metric = mean(metric), .groups = 'drop')
  }
  else if(direction=="min") {
    .data %>%
      group_by(dataset, nr_fold) %>%
      mutate(rank = rank(metric, ties.method = "average")) %>%
      ungroup() %>%
      group_by(dataset, algorithm) %>%
      summarise(average_rank = mean(rank), average_metric = mean(metric), .groups = 'drop')
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

avg_ranks_summarised <- function(.data) {
  .data %>%
    group_by(algorithm) %>%
    summarise(average_rank = mean(average_rank))
} 
