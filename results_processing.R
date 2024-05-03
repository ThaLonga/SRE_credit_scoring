library(pacman)
p_load(tidyverse, knitr, rstatix, tidyposterior, ggplot2)
source("./src/data_loader.R")
source("./src/results_processing_functions.R")
loaded_results <- load_results()
#loaded_results <- loaded_results[-c(4:6)]

combined_results_AUC <- bind_rows(loaded_results$AC_AUC, loaded_results$GC_AUC, loaded_results$HMEQ_AUC, loaded_results$TH02_AUC, loaded_results$LC_AUC, loaded_results$JC_AUC) %>% select(-...1)
combined_results_Brier <- bind_rows(loaded_results$AC_BRIER, loaded_results$GC_BRIER, loaded_results$HMEQ_BRIER, loaded_results$TH02_BRIER, loaded_results$LC_BRIER, loaded_results$JC_BRIER) %>% select(-...1)
combined_results_PG <- bind_rows(loaded_results$AC_PG, loaded_results$GC_PG, loaded_results$HMEQ_PG, loaded_results$TH02_PG, loaded_results$LC_PG, loaded_results$JC_PG) %>% select(-...1)

# AvgRank calculation

average_ranks_AUC <- avg_ranks(combined_results_AUC)
average_ranks_Brier <- avg_ranks(combined_results_Brier, direction = "min")
average_ranks_PG <- avg_ranks(combined_results_PG)

avg_ranks_summarised_AUC <- avg_ranks_summarised(average_ranks_AUC)
avg_ranks_summarised_Brier <- avg_ranks_summarised(average_ranks_Brier)
avg_ranks_summarised_PG <- avg_ranks_summarised(average_ranks_PG)

avg_ranks_summarised_AUC_latex<- xtable(avg_ranks_summarised_AUC)
avg_ranks_summarised_Brier_latex<- xtable(avg_ranks_summarised_Brier)
avg_ranks_summarised_PG_latex<- xtable(avg_ranks_summarised_PG)

#kable(avg_ranks_summarised_AUC, "latex", booktabs = T)

View(avg_ranks_summarised_AUC)
View(average_ranks_AUC)

# Friedman tes
friedman_AUC <- average_ranks_AUC %>% convert_as_factor(dataset, algorithm) %>% select(-average_metric) %>% friedman_test(average_rank ~ algorithm|dataset)
friedman_Brier <- average_ranks_Brier %>% convert_as_factor(dataset, algorithm) %>% friedman_test(average_rank ~ algorithm|dataset)
friedman_PG <- average_ranks_PG %>% convert_as_factor(dataset, algorithm) %>% friedman_test(average_rank ~ algorithm|dataset)


# Bayesian signed rank test (Benavoli et al., 2017)
combined_results_AUC$group <- paste(combined_results_AUC$dataset, combined_results_AUC$nr_fold)
AUC_prep_rank <- combined_results_AUC %>% select(group, algorithm, metric) %>% pivot_wider(names_from = algorithm, values_from = metric) %>% rename("id" = group)
AUC_bayes <- perf_mod(AUC_prep_rank,
                      iter = 20000,
                      seed = 42)
  
combined_results_Brier$group <- paste(combined_results_Brier$dataset, combined_results_Brier$nr_fold)
Brier_prep_rank <- combined_results_Brier %>% select(group, algorithm, metric) %>% pivot_wider(names_from = algorithm, values_from = metric) %>% rename("id" = group)
#Scale between 0.5 and 1
max_Brier <- Brier_prep_rank %>% select(where(is.numeric)) %>% apply(1,max)
Brier_scaled <- cbind(Brier_prep_rank[1],
                     0.5 + 0.5*(Brier_prep_rank[-1]/max_Brier))
Brier_bayes <- perf_mod(Brier_scaled, #NORMALISEREN
                      iter = 20000,
                      seed = 42)

combined_results_PG$group <- paste(combined_results_PG$dataset, combined_results_PG$nr_fold)
PG_prep_rank <- combined_results_PG %>% select(group, algorithm, metric) %>% pivot_wider(names_from = algorithm, values_from = metric) %>% rename("id" = group)
#Scale between 0.5 and 1
max_PG <- PG_prep_rank %>% select(where(is.numeric)) %>% apply(1,max)
PG_scaled <- cbind(PG_prep_rank[1],
                      0.5 + 0.5*(PG_prep_rank[-1]/max_PG))
PG_bayes <- perf_mod(PG_scaled,
                        iter = 20000,
                        seed = 42)

#Most important comparisons: SRE, RE, LRR, RF
contrast_models(AUC_bayes, "SRE", "LRR") %>% ggplot()


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
## CHANGE NAMING FOR 4 DATASETS!!!!!!!!

AUC_results <- do.call(cbind, avg_results[grepl("AUC", names(avg_results))]) %>%
  select(-c(3))
names(AUC_results) <- c("algorithm", "AC", "TH02")
AUC_ranks <- data.frame(apply(AUC_results%>%select(-1), MARGIN = 2, FUN = rank))%>%
  rename(rank_AC = AC) %>%
  rename(rank_TH02 = TH02)
avg_AUC_rank <- rowMeans(AUC_ranks)
AUC_results$avgRank_AUC <- avg_AUC_rank

BRIER_results <- do.call(cbind, avg_results[grepl("BRIER", names(avg_results))]) %>%
  select(-c(3))
names(BRIER_results) <- c("algorithm", "AC", "TH02")
BRIER_ranks <- data.frame(apply(-(BRIER_results%>%select(-1)), MARGIN = 2, FUN = rank))%>%
  rename(rank_AC = AC) %>%
  rename(rank_TH02 = TH02)
avg_BRIER_rank <- rowMeans(BRIER_ranks)
BRIER_results$avgRank_BRIER <- avg_BRIER_rank

PG_results <- do.call(cbind, avg_results[grepl("PG", names(avg_results))]) %>%
  select(-c(3))
names(PG_results) <- c("algorithm", "AC", "TH02")
PG_ranks <- data.frame(apply(PG_results%>%select(-1), MARGIN = 2, FUN = rank))%>%
  rename(rank_AC = AC) %>%
  rename(rank_TH02 = TH02)
avg_PG_rank <- rowMeans(PG_ranks)
PG_results$avgRank_PG <- avg_PG_rank

AUC_results
