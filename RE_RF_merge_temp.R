library(pacman)
p_load(tidyverse, knitr, rstatix, tidyposterior, ggplot2, partykit, xtable)

source("./src/data_loader.R")
source("./src/results_processing_functions.R")
source("./src/adjust_Rom.R")
loaded_results <- load_results()
loaded_results_PLTR <- load_results("results_supp")
loaded_results_DB <- load_results("results_DB")
datasets <- load_data()

combined_results_AUC <- loaded_results[names(loaded_results) %>% grep("v2_AUC_boost_rerun_newpre", .)] %>% 
  bind_rows() %>%
  dplyr::select(-...1) %>% 
  dplyr::filter(algorithm!="XGB")

combined_results_Brier <- loaded_results[names(loaded_results) %>% grep("v2_BRIER_boost_rerun_newpre", .)] %>% 
  bind_rows() %>%
  dplyr::select(-...1)  %>% 
  dplyr::filter(algorithm!="XGB")
combined_results_PG <- loaded_results[names(loaded_results) %>% grep("v2_PG_boost_rerun_newpre", .)] %>% 
  bind_rows() %>%
  dplyr::select(-...1)  %>% 
  dplyr::filter(algorithm!="XGB")
combined_results_EMP <- loaded_results[names(loaded_results) %>% grep("v2_EMP_boost_rerun_newpre", .)] %>% 
  bind_rows() %>%
  dplyr::select(-...1)  %>% 
  dplyr::filter(algorithm!="XGB")


combined_results_AUC_RE_RF <- loaded_results[names(loaded_results) %>% grep("v2_AUC_RE_RF_only", .)] %>% 
  bind_rows() %>%
  dplyr::select(-...1) %>% 
  dplyr::filter(algorithm!="XGB")

combined_results_Brier_RE_RF <- loaded_results[names(loaded_results) %>% grep("v2_BRIER_RE_RF_only", .)] %>% 
  bind_rows() %>%
  dplyr::select(-...1)  %>% 
  dplyr::filter(algorithm!="XGB")
combined_results_PG_RE_RF <- loaded_results[names(loaded_results) %>% grep("v2_PG_RE_RF_only", .)] %>% 
  bind_rows() %>%
  dplyr::select(-...1)  %>% 
  dplyr::filter(algorithm!="XGB")
combined_results_EMP_RE_RF <- loaded_results[names(loaded_results) %>% grep("v2_EMP_RE_RF_only", .)] %>% 
  bind_rows() %>%
  dplyr::select(-...1)  %>% 
  dplyr::filter(algorithm!="XGB")

combined_all_AUC <- dplyr::left_join(combined_results_AUC, combined_results_AUC_RE_RF, by = c("dataset", "nr_fold", "algorithm"))
combined_all_Brier <- dplyr::left_join(combined_results_Brier, combined_results_Brier_RE_RF, by = c("dataset", "nr_fold", "algorithm"))
combined_all_PG <- dplyr::left_join(combined_results_PG, combined_results_PG_RE_RF, by = c("dataset", "nr_fold", "algorithm"))
combined_all_EMP <- dplyr::left_join(combined_results_EMP, combined_results_EMP_RE_RF, by = c("dataset", "nr_fold", "algorithm"))

combined_all_AUC[!is.na(combined_all_AUC$metric.y),]$metric.x <- combined_all_AUC[!is.na(combined_all_AUC$metric.y),]$metric.y
combined_all_Brier[!is.na(combined_all_Brier$metric.y),]$metric.x <- combined_all_Brier[!is.na(combined_all_Brier$metric.y),]$metric.y
combined_all_PG[!is.na(combined_all_PG$metric.y),]$metric.x <- combined_all_PG[!is.na(combined_all_PG$metric.y),]$metric.y
combined_all_EMP[!is.na(combined_all_EMP$metric.y),]$metric.x <- combined_all_EMP[!is.na(combined_all_EMP$metric.y),]$metric.y

combined_all_AUC <- combined_all_AUC%>%dplyr::select(-metric.y)%>%rename(metric = metric.x)
combined_all_Brier <- combined_all_Brier%>%dplyr::select(-metric.y)%>%rename(metric = metric.x)
combined_all_PG <- combined_all_PG%>%dplyr::select(-metric.y)%>%rename(metric = metric.x)
combined_all_EMP <- combined_all_EMP%>%dplyr::select(-metric.y)%>%rename(metric = metric.x)



write_csv(combined_all_AUC, "./results/combined_results_AUC_ORBEL.csv")
write_csv(combined_all_Brier, "./results/combined_results_Brier_ORBEL.csv")
write_csv(combined_all_PG, "./results/combined_results_PG_ORBEL.csv")
write_csv(combined_all_EMP, "./results/combined_results_EMP_ORBEL.csv")
