library(pacman)
p_load(tidyverse)

loaded_results <- load_results()
#loaded_results <- loaded_results[-c(4:6)]
avg_results <- list()
for(name in names(loaded_results)) {
  # calculate means
  avg_results[[name]] <- loaded_results[[name]] %>%
    group_by(algorithm) %>%
    summarise(average = mean(metric))
    
}
dataset_vector <- c("AUC", "BRIER", "PG")
for(name in dataset_vector) {
  avg_results_name <- c(avg_results[grepl(name, names(avg_results))])
  test <- bind_cols(
    avg_results_name[[1]] %>% rename(AC = average), #change
    avg_results_name[[2]][2] %>% rename(TH02 = average)#, change
    #avg_results_name[[3]][2]
  )
  assign(paste0("results_",name), test)
  
  
}

#tables per metric
AUC_results <- do.call(cbind, avg_results[grepl("AUC", names(avg_results))]) %>%
  select(-c(3))
names(AUC_results) <- c("algorithm", "AC", "TH02")

BRIER_results <- do.call(cbind, avg_results[grepl("BRIER", names(avg_results))]) %>%
  select(-c(3))
names(BRIER_results) <- c("algorithm", "AC", "TH02")

PG_results <- do.call(cbind, avg_results[grepl("PG", names(avg_results))]) %>%
  select(-c(3))
names(PG_results) <- c("algorithm", "AC", "TH02")
