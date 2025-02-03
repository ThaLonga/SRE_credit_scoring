library(pacman)
p_load(tidyverse, knitr, rstatix, tidyposterior, ggplot2, partykit, xtable, tidytext)
source("./src/data_loader.R")
source("./src/results_processing_functions.R")
source("./src/adjust_Rom.R")
loaded_results <- load_results()
loaded_results_PLTR <- load_results("results_supp")
loaded_results_DB <- load_results("results_DB")
datasets <- load_data()
nr_datasets = 9


combined_results_AUC <- read_csv("./results/combined_results_AUC_ORBEL_ReSpline.csv") %>% 
  dplyr::filter(algorithm!="SRE_PLTR")
combined_results_Brier <- read_csv("./results/combined_results_Brier_ORBEL_ReSpline.csv") %>% 
  dplyr::filter(algorithm!="SRE_PLTR")
combined_results_PG <- read_csv("./results/combined_results_PG_ORBEL_ReSpline.csv") %>% 
  dplyr::filter(algorithm!="SRE_PLTR")
combined_results_EMP <- read_csv("./results/combined_results_EMP_ORBEL_ReSpline.csv") %>% 
  dplyr::filter(algorithm!="SRE_PLTR")







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

#For no duplicate code
#combined_results_AUC <- combined_results_AUC_DB_config
#combined_results_Brier <- combined_results_Brier_DB_config
#combined_results_PG <- combined_results_PG_DB_config

#########################
# tables for attachments:
#########################
combined_results_AUC_table <- combined_results_AUC %>%
  group_by(dataset, algorithm) %>%
  summarise(avg_metric = round(mean(metric), 3), sd_metric = round(sd(metric), 3)) %>%
  ungroup() %>%
  mutate_if(is.numeric, ~scales::number(., accuracy = 0.001)) 

combined_results_AUC_table$sd_brackets <- mapply(paste, "(", combined_results_AUC_table$sd_metric, ")", MoreArgs = list(sep = ""))
combined_results_AUC_table$AUC <- mapply(paste, combined_results_AUC_table$avg_metric, combined_results_AUC_table$sd_brackets, MoreArgs = list(sep = " "))

finished_AUC_table <- combined_results_AUC_table %>%
  dplyr::select(dataset, algorithm, AUC) %>%
  pivot_wider(names_from = dataset, values_from = AUC)

combined_results_Brier_table <- combined_results_Brier %>%
  group_by(dataset, algorithm) %>%
  summarise(avg_metric = round(mean(metric), 3), sd_metric = round(sd(metric), 3)) %>%
  ungroup() %>%
  mutate_if(is.numeric, ~scales::number(., accuracy = 0.001))

combined_results_Brier_table$sd_brackets <- mapply(paste, "(", combined_results_Brier_table$sd_metric, ")", MoreArgs = list(sep = ""))
combined_results_Brier_table$Brier <- mapply(paste, combined_results_Brier_table$avg_metric, combined_results_Brier_table$sd_brackets, MoreArgs = list(sep = " "))

finished_Brier_table <- combined_results_Brier_table %>%
  dplyr::select(dataset, algorithm, Brier) %>%
  pivot_wider(names_from = dataset, values_from = Brier)

combined_results_PG_table <- combined_results_PG %>%
  group_by(dataset, algorithm) %>%
  summarise(avg_metric = round(mean(metric), 3), sd_metric = round(sd(metric), 3)) %>%
  ungroup() %>%
  mutate_if(is.numeric, ~scales::number(., accuracy = 0.001))

combined_results_PG_table$sd_brackets <- mapply(paste, "(", combined_results_PG_table$sd_metric, ")", MoreArgs = list(sep = ""))
combined_results_PG_table$PG <- mapply(paste, combined_results_PG_table$avg_metric, combined_results_PG_table$sd_brackets, MoreArgs = list(sep = " "))

finished_PG_table <- combined_results_PG_table %>%
  dplyr::select(dataset, algorithm, PG) %>%
  pivot_wider(names_from = dataset, values_from = PG)

combined_results_EMP_table <- combined_results_EMP %>%
  group_by(dataset, algorithm) %>%
  summarise(avg_metric = round(mean(metric), 3), sd_metric = round(sd(metric), 3)) %>%
  ungroup() %>%
  mutate_if(is.numeric, ~scales::number(., accuracy = 0.001))

combined_results_EMP_table$sd_brackets <- mapply(paste, "(", combined_results_EMP_table$sd_metric, ")", MoreArgs = list(sep = ""))
combined_results_EMP_table$EMP <- mapply(paste, combined_results_EMP_table$avg_metric, combined_results_EMP_table$sd_brackets, MoreArgs = list(sep = " "))

finished_EMP_table <- combined_results_EMP_table %>%
  dplyr::select(dataset, algorithm, EMP) %>%
  pivot_wider(names_from = dataset, values_from = EMP)

kable(rbind(finished_AUC_table, finished_Brier_table, finished_PG_table, finished_EMP_table), "latex", booktabs = T)

# compare (S)RE
RE_list <- c("RE_boosting", "RE_RF", "RE_bag", "SRE_boosting", "SRE_RF", "SRE_bag")


#########################
# plots for attachments:
#########################

c15 <- c(
  "brown", "#E31A1C", "#FF7F00", "orange", "gold1",
   
  "orchid1", "darkorchid2", "darkmagenta", 
  "darkblue", "blue1", "dodgerblue3", "deepskyblue3", "darkturquoise","palegreen2","green3",
  
  "white", "white"
)

# AUC plot
combined_results_AUC_plot <- combined_results_AUC%>%
  group_by(dataset, algorithm) %>%
  summarise(avg_metric = round(mean(metric), 3), sd_metric = round(sd(metric), 3)) %>%
  ungroup() %>%
  mutate_if(is.numeric, ~scales::number(., accuracy = 0.001))

combined_results_AUC_plot$algorithm <- factor(combined_results_AUC_plot$algorithm, levels = unique(combined_results_AUC$algorithm))
combined_results_AUC_plot$avg_metric <- as.numeric(combined_results_AUC_plot$avg_metric)

combined_results_AUC_plot <- combined_results_AUC_plot %>%
  add_row(dataset = unique(combined_results_AUC_plot$dataset),
          algorithm = "spacer", avg_metric = 0) %>%
  add_row(dataset = unique(combined_results_AUC_plot$dataset),
          algorithm = "spacer1", avg_metric = 0)

combined_results_AUC_plot$algorithm <- factor(combined_results_AUC_plot$algorithm,
                                              levels = c(
                                                "LRR", "GAM", "LDA", "CTREE", "PLTR", "spacer1", "RF", "XGB", 
                                                "LGBM", "spacer", "RE_boosting", "RE_RF", "RE_bag",
                                                "SRE_RF", "SRE_bag", "SRE_boosting", "SRE_PLTR"
                                              ))



ggplot(combined_results_AUC_plot, aes(x = dataset, y = avg_metric, fill = algorithm)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  labs(x = "Dataset", y = "AUC", fill = "Model") +
  coord_cartesian(xlim = c(NA,NA), ylim = (c(0.5,1))) + 
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    text = element_text(size = 12)
  ) +
  scale_fill_manual(values = c15,
                    breaks = setdiff(levels(combined_results_AUC_plot$algorithm), c("spacer", "spacer1")))

# Brier plot
combined_results_Brier_plot <- combined_results_Brier%>%
  group_by(dataset, algorithm) %>%
  summarise(avg_metric = round(mean(metric), 3), sd_metric = round(sd(metric), 3)) %>%
  ungroup() %>%
  mutate_if(is.numeric, ~scales::number(., accuracy = 0.001))

combined_results_Brier_plot$algorithm <- factor(combined_results_Brier_plot$algorithm, levels = unique(combined_results_Brier$algorithm))
combined_results_Brier_plot$avg_metric <- as.numeric(combined_results_Brier_plot$avg_metric)

combined_results_Brier_plot <- combined_results_Brier_plot %>%
  add_row(dataset = unique(combined_results_Brier_plot$dataset),
          algorithm = "spacer", avg_metric = 0) %>%
  add_row(dataset = unique(combined_results_Brier_plot$dataset),
          algorithm = "spacer1", avg_metric = 0)

combined_results_Brier_plot$algorithm <- factor(combined_results_Brier_plot$algorithm,
                                              levels = c(
                                                "LRR", "GAM", "LDA", "CTREE", "PLTR", "spacer1", "RF", "XGB", 
                                                "LGBM", "spacer", "RE_boosting", "RE_RF", "RE_bag",
                                                "SRE_RF", "SRE_bag", "SRE_boosting", "SRE_PLTR"
                                              ))



ggplot(combined_results_Brier_plot, aes(x = dataset, y = avg_metric, fill = algorithm)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  labs(x = "Dataset", y = "Brier", fill = "Model") +
  coord_cartesian(xlim = c(NA,NA), ylim = (c(0, 0.3))) + 
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    text = element_text(size = 12)
  ) +
  scale_fill_manual(values = c15,
                    breaks = setdiff(levels(combined_results_Brier_plot$algorithm), c("spacer", "spacer1")))

# PG plot
combined_results_PG_plot <- combined_results_PG%>%
  group_by(dataset, algorithm) %>%
  summarise(avg_metric = round(mean(metric), 3), sd_metric = round(sd(metric), 3)) %>%
  ungroup() %>%
  mutate_if(is.numeric, ~scales::number(., accuracy = 0.001))

combined_results_PG_plot$algorithm <- factor(combined_results_PG_plot$algorithm, levels = unique(combined_results_PG$algorithm))
combined_results_PG_plot$avg_metric <- as.numeric(combined_results_PG_plot$avg_metric)

combined_results_PG_plot <- combined_results_PG_plot %>%
  add_row(dataset = unique(combined_results_PG_plot$dataset),
          algorithm = "spacer", avg_metric = 0) %>%
  add_row(dataset = unique(combined_results_PG_plot$dataset),
          algorithm = "spacer1", avg_metric = 0)

combined_results_PG_plot$algorithm <- factor(combined_results_PG_plot$algorithm,
                                                levels = c(
                                                  "LRR", "GAM", "LDA", "CTREE", "PLTR", "spacer1", "RF", "XGB", 
                                                  "LGBM", "spacer", "RE_boosting", "RE_RF", "RE_bag",
                                                  "SRE_RF", "SRE_bag", "SRE_boosting", "SRE_PLTR"
                                                ))



ggplot(combined_results_PG_plot, aes(x = dataset, y = avg_metric, fill = algorithm)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  labs(x = "Dataset", y = "PG", fill = "Model") +
  coord_cartesian(xlim = c(NA,NA), ylim = (c(0, 0.7))) + 
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    text = element_text(size = 12)
  ) +
  scale_fill_manual(values = c15,
                    breaks = setdiff(levels(combined_results_PG_plot$algorithm), c("spacer", "spacer1")))

# EMP plot
combined_results_EMP_plot <- combined_results_EMP%>%
  group_by(dataset, algorithm) %>%
  summarise(avg_metric = round(mean(metric), 3), sd_metric = round(sd(metric), 3)) %>%
  ungroup() %>%
  mutate_if(is.numeric, ~scales::number(., accuracy = 0.001))

combined_results_EMP_plot$algorithm <- factor(combined_results_EMP_plot$algorithm, levels = unique(combined_results_EMP$algorithm))
combined_results_EMP_plot$avg_metric <- as.numeric(combined_results_EMP_plot$avg_metric)

combined_results_EMP_plot <- combined_results_EMP_plot %>%
  add_row(dataset = unique(combined_results_EMP_plot$dataset),
          algorithm = "spacer", avg_metric = 0) %>%
  add_row(dataset = unique(combined_results_EMP_plot$dataset),
          algorithm = "spacer1", avg_metric = 0)

combined_results_EMP_plot$algorithm <- factor(combined_results_EMP_plot$algorithm,
                                             levels = c(
                                               "LRR", "GAM", "LDA", "CTREE", "PLTR", "spacer1", "RF", "XGB", 
                                               "LGBM", "spacer", "RE_boosting", "RE_RF", "RE_bag",
                                               "SRE_RF", "SRE_bag", "SRE_boosting", "SRE_PLTR"
                                             ))



ggplot(combined_results_EMP_plot, aes(x = dataset, y = avg_metric, fill = algorithm)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  labs(x = "Dataset", y = "EMP", fill = "Model") +
  coord_cartesian(xlim = c(NA,NA), ylim = (c(0, 0.1))) + 
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    text = element_text(size = 12)
  ) +
  scale_fill_manual(values = c15,
                    breaks = setdiff(levels(combined_results_EMP_plot$algorithm), c("spacer", "spacer1")))


###############
###############
# All comparisons
###############
###############



if (!require("devtools")) {
  install.packages("devtools")
}

#devtools::install_github("b0rxa/scmamp")

friedman_post_AUC <- scmamp::friedmanPost(AUC_prep_rank[-1], control = "SRE_boosting")
friedman_post_Brier <- scmamp::friedmanPost(Brier_prep_rank[-1], control = "SRE_boosting")
friedman_post_PG <- scmamp::friedmanPost(PG_prep_rank[-1], control = "LRR")
friedman_post_EMP <- scmamp::friedmanPost(EMP_prep_rank[-1], control = "SRE_RF")

friedman_post_AUC_corrected <- adjustRom(friedman_post_AUC)
friedman_post_Brier_corrected <- adjustRom(friedman_post_Brier)
friedman_post_PG_corrected <- adjustRom(friedman_post_PG)
friedman_post_EMP_corrected <- adjustRom(friedman_post_EMP)

friedman_post_AUC_corrected[is.na(friedman_post_AUC_corrected)] <- 1.0
friedman_post_Brier_corrected[is.na(friedman_post_Brier_corrected)] <- 1.0
friedman_post_PG_corrected[is.na(friedman_post_PG_corrected)] <- 1.0
friedman_post_EMP_corrected[is.na(friedman_post_EMP_corrected)] <- 1.0


friedman_data_AUC <- combined_results_AUC %>%
  select(dataset, nr_fold, algorithm, metric) %>%
  pivot_wider(names_from = algorithm, values_from = metric)
metric_matrix_AUC <- as.matrix(friedman_data_AUC[, -c(1, 2)])
friedman_result_AUC <- friedman.test(metric_matrix_AUC)

friedman_data_Brier <- combined_results_Brier %>%
  select(dataset, nr_fold, algorithm, metric) %>%
  pivot_wider(names_from = algorithm, values_from = metric)
metric_matrix_Brier <- as.matrix(friedman_data_Brier[, -c(1, 2)])
friedman_result_Brier <- friedman.test(metric_matrix_Brier)

friedman_data_PG <- combined_results_PG %>%
  select(dataset, nr_fold, algorithm, metric) %>%
  pivot_wider(names_from = algorithm, values_from = metric)
metric_matrix_PG <- as.matrix(friedman_data_PG[, -c(1, 2)])
friedman_result_PG <- friedman.test(metric_matrix_PG)

friedman_data_EMP <- combined_results_EMP %>%
  select(dataset, nr_fold, algorithm, metric) %>%
  pivot_wider(names_from = algorithm, values_from = metric)
metric_matrix_EMP <- as.matrix(friedman_data_EMP[, -c(1, 2)])
friedman_result_EMP <- friedman.test(metric_matrix_EMP)

print(friedman_result_EMP)









#########################
# AvgRank calculation
#########################

average_ranks_AUC <- avg_ranks(combined_results_AUC)
average_ranks_Brier <- avg_ranks(combined_results_Brier, direction = "min")
average_ranks_PG <- avg_ranks(combined_results_PG)
average_ranks_EMP <- avg_ranks(combined_results_EMP)

avg_ranks_summarized_AUC <- avg_ranks_summarized(average_ranks_AUC)
avg_ranks_summarized_Brier <- avg_ranks_summarized(average_ranks_Brier)
avg_ranks_summarized_PG <- avg_ranks_summarized(average_ranks_PG)
avg_ranks_summarized_EMP <- avg_ranks_summarized(average_ranks_EMP)

avg_ranks_summarized_AUC_latex<- xtable(avg_ranks_summarized_AUC)
avg_ranks_summarized_Brier_latex<- xtable(avg_ranks_summarized_Brier)
avg_ranks_summarized_PG_latex<- xtable(avg_ranks_summarized_PG)
avg_ranks_summarized_EMP_latex<- xtable(avg_ranks_summarized_EMP)

#kable(avg_ranks_summarized_AUC, "latex", booktabs = T)

View(avg_ranks_summarized_EMP)
View(average_ranks_AUC)


###############
# Friedman test
###############

combined_results_AUC$group <- paste(combined_results_AUC$dataset, combined_results_AUC$nr_fold)
AUC_prep_rank <- combined_results_AUC %>% dplyr::select(group, algorithm, metric) %>% pivot_wider(names_from = algorithm, values_from = metric) %>% rename("id" = group)

combined_results_Brier$group <- paste(combined_results_Brier$dataset, combined_results_Brier$nr_fold)
Brier_prep_rank <- combined_results_Brier %>% dplyr::select(group, algorithm, metric) %>% pivot_wider(names_from = algorithm, values_from = metric) %>% rename("id" = group)

combined_results_PG$group <- paste(combined_results_PG$dataset, combined_results_PG$nr_fold)
PG_prep_rank <- combined_results_PG %>% dplyr::select(group, algorithm, metric) %>% pivot_wider(names_from = algorithm, values_from = metric) %>% rename("id" = group)

combined_results_EMP$group <- paste(combined_results_EMP$dataset, combined_results_EMP$nr_fold)
EMP_prep_rank <- combined_results_EMP %>% dplyr::select(group, algorithm, metric) %>% pivot_wider(names_from = algorithm, values_from = metric) %>% rename("id" = group)

#friedman_AUC <- average_ranks_AUC %>% convert_as_factor(dataset, algorithm) %>% dplyr::select(-average_metric) %>% friedman_test(average_rank ~ algorithm|dataset)
#friedman_Brier <- average_ranks_Brier %>% convert_as_factor(dataset, algorithm) %>% friedman_test(average_rank ~ algorithm|dataset)
#friedman_PG <- average_ranks_PG %>% convert_as_factor(dataset, algorithm) %>% friedman_test(average_rank ~ algorithm|dataset)
#friedman_EMP <- average_ranks_EMP %>% convert_as_factor(dataset, algorithm) %>% friedman_test(average_rank ~ algorithm|dataset)


#AUC pairwise friedman
#AUC_pairwise_p_values <- c()
#for(i in 1:nrow(avg_ranks_summarized_AUC)) {
#  R_j <- min(avg_ranks_summarized_AUC$average_rank)
#  z <- friedman_pairwise(R_j, avg_ranks_summarized_AUC$average_rank[i], N = nr_datasets, k = nrow(avg_ranks_summarized_AUC))
#  AUC_pairwise_p_values[i] <- pnorm(z, lower.tail = FALSE)*2
#}
#AUC_pairwise_p_values_adjusted <- adjustRom(AUC_pairwise_p_values, alpha=0.05)
#
##Brier pairwise friedman
#Brier_pairwise_p_values <- c()
#for(i in 1:nrow(avg_ranks_summarized_Brier)) {
#  R_j <- min(avg_ranks_summarized_Brier$average_rank)
#  z <- friedman_pairwise(R_j, avg_ranks_summarized_Brier$average_rank[i], N = nr_datasets, k = nrow(avg_ranks_summarized_Brier))
#  Brier_pairwise_p_values[i] <- pnorm(z, lower.tail = FALSE)*2
#}
#Brier_pairwise_p_values_adjusted <- adjustRom(Brier_pairwise_p_values, alpha=0.05)
#
##PG pairwise friedman
#PG_pairwise_p_values <- c()
#for(i in 1:nrow(avg_ranks_summarized_PG)) {
#  R_j <- min(avg_ranks_summarized_PG$average_rank)
#  z <- friedman_pairwise(R_j, avg_ranks_summarized_PG$average_rank[i], N = nr_datasets, k = nrow(avg_ranks_summarized_PG))
#  PG_pairwise_p_values[i] <- pnorm(z, lower.tail = FALSE)*2
#}
#PG_pairwise_p_values_adjusted <- adjustRom(PG_pairwise_p_values, alpha=0.05)
#
##EMP pairwise friedman
#EMP_pairwise_p_values <- c()
#for(i in 1:nrow(avg_ranks_summarized_EMP)) {
#  R_j <- min(avg_ranks_summarized_EMP$average_rank)
#  z <- friedman_pairwise(R_j, avg_ranks_summarized_EMP$average_rank[i], N = nr_datasets, k = nrow(avg_ranks_summarized_EMP))
#  EMP_pairwise_p_values[i] <- pnorm(z, lower.tail = FALSE)*2
#}
#EMP_pairwise_p_values_adjusted <- adjustRom(EMP_pairwise_p_values, alpha=0.05)

#join to make table
all_avg_ranks <- cbind(avg_ranks_summarized_AUC$algorithm, round(avg_ranks_summarized_AUC$average_rank, 2), round(avg_ranks_summarized_Brier$average_rank, 2), round(avg_ranks_summarized_PG$average_rank, 2), round(avg_ranks_summarized_EMP$average_rank, 2)) %>% as_tibble()
pairwise_p_values <- cbind(c(round(friedman_post_AUC_corrected , 3)), c(round(friedman_post_Brier_corrected , 3)), c(round(friedman_post_PG_corrected , 3)), c(round(friedman_post_EMP_corrected, 3))) %>% as_tibble()
pairwise_p_values_brackets <- as.data.frame(mapply(paste, "(", pairwise_p_values, ")", MoreArgs = list(sep = "")))
pairwise_p_values_brackets <- cbind(colnames(friedman_post_AUC), pairwise_p_values_brackets)
#pairwise_p_values_brackets[1]<-NA
colnames(pairwise_p_values_brackets) <- c("Algorithm", "AUC", "Brier", "PG", "EMP")

# order the algorithms
order_vector <- c("LRR", "GAM", "LDA", "CTREE", "RF", "LGBM", "PLTR", "RE_boosting", 
                  "RE_RF", "RE_bag", "SRE_RF", "SRE_bag", "SRE_boosting", 
                  "SRE_PLTR")

all_avg_ranks_ordered <- all_avg_ranks %>%
  mutate(V1 = factor(V1, levels = order_vector)) %>%
  arrange(V1)

pairwise_p_values_brackets_ordered <- pairwise_p_values_brackets %>%
  mutate(Algorithm = factor(Algorithm, levels = order_vector)) %>%
  arrange(Algorithm)
pairwise_p_values_brackets_ordered[1] <- NA

table_pvalues <- as.tibble(mapply(paste, all_avg_ranks_ordered, pairwise_p_values_brackets_ordered, MoreArgs = list(sep = " ")))
table_pvalues <- as.tibble(lapply(table_pvalues, function(x) {
  gsub(" NA", "", x)
}))
colnames(table_pvalues) <- c("Algorithm", "AUC", "Brier", "PG", "EMP")
table_pvalues_latex <- format_p_values(kable(table_pvalues, "latex", booktabs = T))

###########################################################################
###########################################################################
# Stepwise comparisons
###########################################################################
###########################################################################
# RE -> select best -> compare with interpretable & compare with explainable

interpretable_list <- c("LRR", "GAM", "LDA", "CTREE", "PLTR")
explainable_list <- c("RF", "XGB", "LGBM")
benchmark_list_AUC_Brier_EMP <- c("LRR", "GAM", "LDA", "CTREE", "RF", "LGBM", "PLTR", "SRE_boosting")
benchmark_list_PG <- c("LRR", "GAM", "LDA", "CTREE", "RF", "LGBM", "PLTR", "SRE_RF")


#########################
# RE
#########################

friedman_post_AUC <- scmamp::friedmanPost(AUC_prep_rank[-1]%>%select(all_of(RE_list)), control = "SRE_boosting")
friedman_post_Brier <- scmamp::friedmanPost(Brier_prep_rank[-1]%>%select(all_of(RE_list)), control = "SRE_boosting")
friedman_post_PG <- scmamp::friedmanPost(PG_prep_rank[-1]%>%select(all_of(RE_list)), control = "SRE_RF")
friedman_post_EMP <- scmamp::friedmanPost(EMP_prep_rank[-1]%>%select(all_of(RE_list)), control = "SRE_boosting")

friedman_post_AUC_corrected <- adjustRom(friedman_post_AUC)
friedman_post_Brier_corrected <- adjustRom(friedman_post_Brier)
friedman_post_PG_corrected <- adjustRom(friedman_post_PG)
friedman_post_EMP_corrected <- adjustRom(friedman_post_EMP)

#friedman_post_AUC_corrected[is.na(friedman_post_AUC_corrected)] <- 1.0
#friedman_post_Brier_corrected[is.na(friedman_post_Brier_corrected)] <- 1.0
#friedman_post_PG_corrected[is.na(friedman_post_PG_corrected)] <- 1.0
#friedman_post_EMP_corrected[is.na(friedman_post_EMP_corrected)] <- 1.0


friedman_data_AUC <- combined_results_AUC %>%
  select(dataset, nr_fold, algorithm, metric) %>%
  pivot_wider(names_from = algorithm, values_from = metric)
metric_matrix_AUC <- as.matrix(friedman_data_AUC[, -c(1, 2)])
friedman_result_AUC <- friedman.test(metric_matrix_AUC)

friedman_data_Brier <- combined_results_Brier %>%
  select(dataset, nr_fold, algorithm, metric) %>%
  pivot_wider(names_from = algorithm, values_from = metric)
metric_matrix_Brier <- as.matrix(friedman_data_Brier[, -c(1, 2)])
friedman_result_Brier <- friedman.test(metric_matrix_Brier)

friedman_data_PG <- combined_results_PG %>%
  select(dataset, nr_fold, algorithm, metric) %>%
  pivot_wider(names_from = algorithm, values_from = metric)
metric_matrix_PG <- as.matrix(friedman_data_PG[, -c(1, 2)])
friedman_result_PG <- friedman.test(metric_matrix_PG)

friedman_data_EMP <- combined_results_EMP %>%
  select(dataset, nr_fold, algorithm, metric) %>%
  pivot_wider(names_from = algorithm, values_from = metric)
metric_matrix_EMP <- as.matrix(friedman_data_EMP[, -c(1, 2)])
friedman_result_EMP <- friedman.test(metric_matrix_EMP)

average_ranks_RE_AUC <- avg_ranks(combined_results_AUC%>%filter(algorithm %in% RE_list))
average_ranks_RE_Brier <- avg_ranks(combined_results_Brier%>%filter(algorithm %in% RE_list), direction = "min")
average_ranks_RE_PG <- avg_ranks(combined_results_PG%>%filter(algorithm %in% RE_list))
average_ranks_RE_EMP <- avg_ranks(combined_results_EMP%>%filter(algorithm %in% RE_list))

avg_ranks_RE_summarized_AUC <- avg_ranks_summarized(average_ranks_RE_AUC)
avg_ranks_RE_summarized_Brier <- avg_ranks_summarized(average_ranks_RE_Brier)
avg_ranks_RE_summarized_PG <- avg_ranks_summarized(average_ranks_RE_PG)
avg_ranks_RE_summarized_EMP <- avg_ranks_summarized(average_ranks_RE_EMP)


all_avg_ranks <- cbind(avg_ranks_RE_summarized_AUC$algorithm, round(avg_ranks_RE_summarized_AUC$average_rank, 2), round(avg_ranks_RE_summarized_Brier$average_rank, 2), round(avg_ranks_RE_summarized_PG$average_rank, 2), round(avg_ranks_RE_summarized_EMP$average_rank, 2)) %>% as_tibble()
pairwise_p_values <- cbind(c(round(friedman_post_AUC_corrected , 3)), c(round(friedman_post_Brier_corrected , 3)), c(round(friedman_post_PG_corrected , 3)), c(round(friedman_post_EMP_corrected, 3))) %>% as_tibble()
pairwise_p_values_brackets <- as.data.frame(mapply(paste, "(", pairwise_p_values, ")", MoreArgs = list(sep = "")))
pairwise_p_values_brackets <- cbind(colnames(friedman_post_AUC), pairwise_p_values_brackets)
#pairwise_p_values_brackets[1]<-NA
colnames(pairwise_p_values_brackets) <- c("Algorithm", "AUC", "Brier", "PG", "EMP")

# order the algorithms
order_vector <- c("RE_boosting", "RE_RF", "RE_bag", "SRE_RF", "SRE_bag", "SRE_boosting")

all_avg_ranks_ordered <- all_avg_ranks %>%
  mutate(V1 = factor(V1, levels = order_vector)) %>%
  arrange(V1)

pairwise_p_values_brackets_ordered <- pairwise_p_values_brackets %>%
  mutate(Algorithm = factor(Algorithm, levels = order_vector)) %>%
  arrange(Algorithm)
pairwise_p_values_brackets_ordered[1] <- NA

table_pvalues <- as.tibble(mapply(paste, all_avg_ranks_ordered, pairwise_p_values_brackets_ordered, MoreArgs = list(sep = " ")))
table_pvalues <- as.tibble(lapply(table_pvalues, function(x) {
  gsub(" NA", "", x)
}))
colnames(table_pvalues) <- c("Algorithm", "AUC", "Brier", "PG", "EMP")
table_pvalues_latex <- format_p_values(kable(table_pvalues, "latex", booktabs = T))
colnames(all_avg_ranks) <- c("Algorithm", "AUC", "Brier", "PG", "EMP")
table_RE_ranks_latex <- (kable(all_avg_ranks, "latex", booktabs = T))




rank_plot <- all_avg_ranks
rank_plot <- rank_plot %>% mutate(V2 = as.numeric(V2), V3 = as.numeric(V3), V4 = as.numeric(V4), V5 = as.numeric(V5))
colnames(rank_plot) <- c("Algorithm", "AUC", "Brier", "PG", "EMP")
rank_plot_long <- rank_plot %>% 
  pivot_longer(cols = -Algorithm, names_to = "Metric", values_to = "Rank") %>%
  group_by(Metric) %>%
  arrange(desc(Rank), .by_group = TRUE)

ggplot(rank_plot_long, aes(x = reorder_within(Algorithm, Rank, Metric), 
                           y = Rank, 
                           fill = Algorithm)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_text(aes(label = round(Rank, 2)),  # Add the rank values as labels
            hjust = 1.3,                # Adjust text vertically (above bars)
            size = 3,                    # Text size
            angle = 90) +                # Rotate text vertically
  facet_wrap(~ Metric, scales = "free_x", ncol = 2) +
  scale_x_reordered() +  # Automatically adjust x-axis labels for each facet
  scale_fill_brewer(palette = "Dark2") +  # Choose a color palette
  theme_minimal() +
  theme(
    strip.text = element_text(size = 12, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.title = element_text(size = 12),
    legend.position = "none"
  ) +
  labs(
    x = "Algorithm",
    y = "Rank",
    title = "Rank of algorithms by metric",
  )



ggplot(rank_plot_long %>% 
         group_by(Metric) %>% 
         mutate(is_best = Rank == min(Rank)), 
       aes(x = Rank, 
           y = reorder_within(Algorithm, -Rank, Metric))) + # Reverse Rank for ascending order
  # Add stems with conditional coloring
  geom_segment(aes(x = 0, 
                   xend = Rank, 
                   y = reorder_within(Algorithm, -Rank, Metric), 
                   yend = reorder_within(Algorithm, -Rank, Metric), 
                   color = ifelse(is_best, "Best", "Other")), 
               size = 0.8) +
  # Add points (remove black edge with color = NA)
  geom_point(
    aes(fill = ifelse(is_best, "Best", "Other"),
        color = ifelse(is_best, "Best", "Other"),
        size = ifelse(is_best, 3, 3)),
    shape = 21) +
  # Facet by Metric to visually group
  facet_wrap(~ Metric, scales = "free_y", ncol = 2) +
  # Manual color scales (grey for others, orange for best)
  scale_fill_manual(values = c("Best" = "orange", "Other" = "grey"), guide = "none") +
  scale_color_manual(values = c("Best" = "orange", "Other" = "grey"), guide = "none") +
  # Adjust size for points
  scale_size_identity() +
  scale_y_reordered() +
  
  # Add labels and theme adjustments
  labs(x = "", y = "") +
  theme_minimal() +
  theme(
    strip.text = element_text(face = "bold", size = 11) # Make facet titles bold
  )







#########################
# SRE vs benchmark
#########################


friedman_AUC <- average_ranks_AUC %>% filter(algorithm %in% benchmark_list_AUC_Brier_EMP) %>% convert_as_factor(dataset, algorithm)  %>% friedman_test(average_rank ~ algorithm|dataset)
friedman_Brier <- average_ranks_Brier %>% filter(algorithm %in%benchmark_list_AUC_Brier_EMP) %>% convert_as_factor(dataset, algorithm) %>% friedman_test(average_rank ~ algorithm|dataset)
friedman_PG <- average_ranks_PG %>% filter(algorithm %in%benchmark_list_PG) %>% convert_as_factor(dataset, algorithm) %>% friedman_test(average_rank ~ algorithm|dataset)
friedman_EMP <- average_ranks_EMP %>% filter(algorithm %in%benchmark_list_AUC_Brier_EMP) %>% convert_as_factor(dataset, algorithm) %>% friedman_test(average_rank ~ algorithm|dataset)


friedman_post_AUC <- scmamp::friedmanPost(AUC_prep_rank[-1]%>%select(all_of(benchmark_list_AUC_Brier_EMP)), control = "SRE_boosting")
friedman_post_Brier <- scmamp::friedmanPost(Brier_prep_rank[-1]%>%select(all_of(benchmark_list_AUC_Brier_EMP)), control = "SRE_boosting")
friedman_post_PG <- scmamp::friedmanPost(PG_prep_rank[-1]%>%select(all_of(benchmark_list_PG)), control = "SRE_RF")
friedman_post_EMP <- scmamp::friedmanPost(EMP_prep_rank[-1]%>%select(all_of(benchmark_list_AUC_Brier_EMP)), control = "SRE_boosting")

friedman_post_AUC_corrected <- adjustRom(friedman_post_AUC)
friedman_post_Brier_corrected <- adjustRom(friedman_post_Brier)
friedman_post_PG_corrected <- adjustRom(friedman_post_PG)
friedman_post_EMP_corrected <- adjustRom(friedman_post_EMP)

friedman_data_AUC <- combined_results_AUC %>%
  select(dataset, nr_fold, algorithm, metric) %>%
  pivot_wider(names_from = algorithm, values_from = metric)
metric_matrix_AUC <- as.matrix(friedman_data_AUC[, -c(1, 2)])
friedman_result_AUC <- friedman.test(metric_matrix_AUC)

friedman_data_Brier <- combined_results_Brier %>%
  select(dataset, nr_fold, algorithm, metric) %>%
  pivot_wider(names_from = algorithm, values_from = metric)
metric_matrix_Brier <- as.matrix(friedman_data_Brier[, -c(1, 2)])
friedman_result_Brier <- friedman.test(metric_matrix_Brier)

friedman_data_PG <- combined_results_PG %>%
  select(dataset, nr_fold, algorithm, metric) %>%
  pivot_wider(names_from = algorithm, values_from = metric)
metric_matrix_PG <- as.matrix(friedman_data_PG[, -c(1, 2)])
friedman_result_PG <- friedman.test(metric_matrix_PG)

friedman_data_EMP <- combined_results_EMP %>%
  select(dataset, nr_fold, algorithm, metric) %>%
  pivot_wider(names_from = algorithm, values_from = metric)
metric_matrix_EMP <- as.matrix(friedman_data_EMP[, -c(1, 2)])
friedman_result_EMP <- friedman.test(metric_matrix_EMP)

average_ranks_benchmark_AUC <- avg_ranks(combined_results_AUC%>%filter(algorithm %in% benchmark_list_AUC_Brier_EMP))
average_ranks_benchmark_Brier <- avg_ranks(combined_results_Brier%>%filter(algorithm %in% benchmark_list_AUC_Brier_EMP), direction = "min")
average_ranks_benchmark_PG <- avg_ranks(combined_results_PG%>%filter(algorithm %in% benchmark_list_PG))
average_ranks_benchmark_EMP <- avg_ranks(combined_results_EMP%>%filter(algorithm %in% benchmark_list_AUC_Brier_EMP))

avg_ranks_benchmark_summarized_AUC <- avg_ranks_summarized(average_ranks_benchmark_AUC)
avg_ranks_benchmark_summarized_Brier <- avg_ranks_summarized(average_ranks_benchmark_Brier)
avg_ranks_benchmark_summarized_PG <- avg_ranks_summarized(average_ranks_benchmark_PG)
avg_ranks_benchmark_summarized_EMP <- avg_ranks_summarized(average_ranks_benchmark_EMP)


all_avg_ranks_ABE <- cbind(avg_ranks_benchmark_summarized_AUC$algorithm, round(avg_ranks_benchmark_summarized_AUC$average_rank, 2), round(avg_ranks_benchmark_summarized_Brier$average_rank, 2), round(avg_ranks_benchmark_summarized_PG$average_rank, 2), round(avg_ranks_benchmark_summarized_EMP$average_rank, 2)) %>% as_tibble()
all_avg_ranks_ABE[8, 4] <- NA
all_avg_ranks_PG <- cbind(avg_ranks_benchmark_summarized_EMP$algorithm, round(avg_ranks_benchmark_summarized_AUC$average_rank, 2), round(avg_ranks_benchmark_summarized_Brier$average_rank, 2), round(avg_ranks_benchmark_summarized_PG$average_rank, 2), round(avg_ranks_benchmark_summarized_EMP$average_rank, 2)) %>% as_tibble()
all_avg_ranks_PG[8, c(2:3, 5)] <- NA
pairwise_p_values <- cbind(c(round(friedman_post_AUC_corrected , 3)), c(round(friedman_post_Brier_corrected , 3)), c(round(friedman_post_PG_corrected , 3)), c(round(friedman_post_EMP_corrected, 3))) %>% as_tibble()
pairwise_p_values_brackets <- as.data.frame(mapply(paste, "(", pairwise_p_values, ")", MoreArgs = list(sep = "")))
pairwise_p_values_brackets <- cbind(colnames(friedman_post_AUC), pairwise_p_values_brackets)
#pairwise_p_values_brackets[1]<-NA
colnames(pairwise_p_values_brackets) <- c("Algorithm", "AUC", "Brier", "PG", "EMP")
pairwise_p_values_brackets <- rbind(pairwise_p_values_brackets, c("SRE_RF", "(NA)", "(NA)", "(NA)", "(NA)"))

# order the algorithms
order_vector <- c("LRR", "GAM", "LDA", "CTREE", "RF", "LGBM", "PLTR", "SRE_boosting", "SRE_RF")

all_avg_ranks_ordered_ABE <- all_avg_ranks_ABE %>%
  mutate(V1 = factor(V1, levels = order_vector)) %>%
  arrange(V1)

all_avg_ranks_ordered <- rbind(all_avg_ranks_ordered_ABE, all_avg_ranks_PG[8,])

pairwise_p_values_brackets_ordered <- pairwise_p_values_brackets %>%
  mutate(Algorithm = factor(Algorithm, levels = order_vector)) %>%
  arrange(Algorithm)
pairwise_p_values_brackets_ordered[1] <- NA

table_pvalues <- as.tibble(mapply(paste, all_avg_ranks_ordered, pairwise_p_values_brackets_ordered, MoreArgs = list(sep = " ")))
table_pvalues <- as.tibble(lapply(table_pvalues, function(x) {
  gsub(" NA", "", x)
}))
colnames(table_pvalues) <- c("Algorithm", "AUC", "Brier", "PG", "EMP")
table_pvalues_latex <- format_p_values(kable(table_pvalues, "latex", booktabs = T))

rank_plot <- all_avg_ranks_ABE
rank_plot[is.na(rank_plot)] <- "3.44"
rank_plot <- rank_plot %>% mutate(V2 = as.numeric(V2), V3 = as.numeric(V3), V4 = as.numeric(V4), V5 = as.numeric(V5))
colnames(rank_plot) <- c("Algorithm", "AUC", "Brier", "PG", "EMP")
rank_plot[8,1] <- "SRE"
rank_plot_long <- rank_plot %>% 
  pivot_longer(cols = -Algorithm, names_to = "Metric", values_to = "Rank") %>%
  group_by(Metric) %>%
  arrange(desc(Rank), .by_group = TRUE)


rank_plot_long$Metric[rank_plot_long$Metric=="Brier"] <- "Brier_Score"
rank_plot_long$Metric[rank_plot_long$Metric=="PG"] <- "Partial Gini"




# benchmark
ggplot(rank_plot_long %>% 
         group_by(Metric) %>% 
         mutate(is_best = Rank == min(Rank)), 
       aes(x = Rank, 
           y = reorder_within(Algorithm, -Rank, Metric))) + # Reverse Rank for ascending order
  # Add stems with conditional coloring
  geom_segment(aes(x = 0, 
                   xend = Rank, 
                   y = reorder_within(Algorithm, -Rank, Metric), 
                   yend = reorder_within(Algorithm, -Rank, Metric), 
                   color = case_when(
                     Algorithm == "SRE" ~ "SRE",
                     Algorithm == "LRR" ~ "LRR",
                     TRUE ~ "Other"
                   )), 
               size = 0.8) +
  # Add points (remove black edge with color = NA)
  geom_point(
    aes(fill = case_when(
      Algorithm == "SRE" ~ "SRE",
      Algorithm == "LRR" ~ "LRR",
      TRUE ~ "Other"),
        color = case_when(
          Algorithm == "SRE" ~ "SRE",
          Algorithm == "LRR" ~ "LRR",
          TRUE ~ "Other"
        ),
        size = ifelse(is_best, 3, 3)),
    shape = 21) +
  # Facet by Metric to visually group
  facet_wrap(~ Metric, scales = "free_y", ncol = 2) +
  # Manual color scales (grey for others, orange for best)
  scale_fill_manual(values = c("SRE" = "orange", "LRR" = "black", "Other" = "grey"), guide = "none") +
  scale_color_manual(values = c("SRE" = "orange", "LRR" = "black", "Other" = "grey"), guide = "none") +
  # Adjust size for points
  scale_size_identity() +
  scale_y_reordered() +
  
  # Add labels and theme adjustments
  labs(x = "", y = "") +
  theme_minimal() +
  theme(
    strip.text = element_text(face = "bold", size = 11) # Make facet titles bold
  )
  
  
  




#########################
# interpretable
#########################

# AvgRank calculation
average_ranks_I_AUC <- avg_ranks(combined_results_AUC%>%filter(algorithm %in% c(interpretable_list, "SRE_boosting")))
average_ranks_I_Brier <- avg_ranks(combined_results_Brier%>%filter(algorithm %in% c(interpretable_list, "SRE_boosting")), direction = "min")
average_ranks_I_PG <- avg_ranks(combined_results_PG%>%filter(algorithm %in% c(interpretable_list, "SRE_boosting")))
average_ranks_I_EMP <- avg_ranks(combined_results_EMP%>%filter(algorithm %in% c(interpretable_list, "SRE_boosting")))

avg_ranks_I_summarized_AUC <- avg_ranks_summarized(average_ranks_I_AUC)
avg_ranks_I_summarized_Brier <- avg_ranks_summarized(average_ranks_I_Brier)
avg_ranks_I_summarized_PG <- avg_ranks_summarized(average_ranks_I_PG)
avg_ranks_I_summarized_EMP <- avg_ranks_summarized(average_ranks_I_EMP)

avg_ranks_I_summarized_AUC_latex<- xtable(avg_ranks_I_summarized_AUC)
avg_ranks_I_summarized_Brier_latex<- xtable(avg_ranks_I_summarized_Brier)
avg_ranks_I_summarized_PG_latex<- xtable(avg_ranks_I_summarized_PG)
avg_ranks_I_summarized_EMP_latex<- xtable(avg_ranks_I_summarized_EMP)

#kable(avg_ranks_summarized_AUC, "latex", booktabs = T)


###############
# Friedman test
###############

friedman_I_AUC <- average_ranks_I_AUC %>% convert_as_factor(dataset, algorithm) %>% dplyr::select(-average_metric) %>% friedman_test(average_rank ~ algorithm|dataset)
friedman_I_Brier <- average_ranks_I_Brier %>% convert_as_factor(dataset, algorithm) %>% friedman_test(average_rank ~ algorithm|dataset)
friedman_I_PG <- average_ranks_I_PG %>% convert_as_factor(dataset, algorithm) %>% friedman_test(average_rank ~ algorithm|dataset)
friedman_I_EMP <- average_ranks_I_EMP %>% convert_as_factor(dataset, algorithm) %>% friedman_test(average_rank ~ algorithm|dataset)


#AUC pairwise friedman
I_AUC_pairwise_p_values <- c()
for(i in 1:nrow(avg_ranks_I_summarized_AUC)) {
  R_j <- min(avg_ranks_I_summarized_AUC$average_rank)
  z <- friedman_pairwise(R_j, avg_ranks_I_summarized_AUC$average_rank[i], N = nr_datasets, k = nrow(avg_ranks_I_summarized_AUC))
  I_AUC_pairwise_p_values[i] <- pnorm(z, lower.tail = FALSE)*2
}
I_AUC_pairwise_p_values_adjusted <- adjustRom(I_AUC_pairwise_p_values, alpha=0.05)

#Brier pairwise friedman
I_Brier_pairwise_p_values <- c()
for(i in 1:nrow(avg_ranks_I_summarized_Brier)) {
  R_j <- min(avg_ranks_I_summarized_Brier$average_rank)
  z <- friedman_pairwise(R_j, avg_ranks_I_summarized_Brier$average_rank[i], N = nr_datasets, k = nrow(avg_ranks_I_summarized_Brier))
  I_Brier_pairwise_p_values[i] <- pnorm(z, lower.tail = FALSE)*2
}
I_Brier_pairwise_p_values_adjusted <- adjustRom(I_Brier_pairwise_p_values, alpha=0.05)

#PG pairwise friedman
I_PG_pairwise_p_values <- c()
for(i in 1:nrow(avg_ranks_I_summarized_PG)) {
  R_j <- min(avg_ranks_I_summarized_PG$average_rank)
  z <- friedman_pairwise(R_j, avg_ranks_I_summarized_PG$average_rank[i], N = nr_datasets, k = nrow(avg_ranks_I_summarized_PG))
  I_PG_pairwise_p_values[i] <- pnorm(z, lower.tail = FALSE)*2
}
I_PG_pairwise_p_values_adjusted <- adjustRom(I_PG_pairwise_p_values, alpha=0.05)

#EMP pairwise friedman
I_EMP_pairwise_p_values <- c()
for(i in 1:nrow(avg_ranks_I_summarized_EMP)) {
  R_j <- min(avg_ranks_I_summarized_EMP$average_rank)
  z <- friedman_pairwise(R_j, avg_ranks_I_summarized_EMP$average_rank[i], N = nr_datasets, k = nrow(avg_ranks_I_summarized_EMP))
  I_EMP_pairwise_p_values[i] <- pnorm(z, lower.tail = FALSE)*2
}
I_EMP_pairwise_p_values_adjusted <- adjustRom(I_EMP_pairwise_p_values, alpha=0.05)

#join to make table
I_all_avg_ranks <- cbind(avg_ranks_I_summarized_AUC$algorithm, round(avg_ranks_I_summarized_AUC$average_rank, 2), round(avg_ranks_I_summarized_Brier$average_rank, 2), round(avg_ranks_I_summarized_PG$average_rank, 2), round(avg_ranks_I_summarized_EMP$average_rank, 2)) %>% as_tibble()
I_pairwise_p_values <- cbind(avg_ranks_I_summarized_AUC$algorithm, round(I_AUC_pairwise_p_values_adjusted, 3), round(I_Brier_pairwise_p_values_adjusted, 3), round(I_PG_pairwise_p_values_adjusted, 3), round(I_EMP_pairwise_p_values_adjusted, 3)) %>% as_tibble()
I_pairwise_p_values_brackets <- as.data.frame(mapply(paste, "(", I_pairwise_p_values, ")", MoreArgs = list(sep = "")))
I_pairwise_p_values_brackets[1]<-NA

I_table_pvalues <- as.tibble(mapply(paste, I_all_avg_ranks, I_pairwise_p_values_brackets, MoreArgs = list(sep = " ")))
I_table_pvalues <- as.tibble(lapply(I_table_pvalues, function(x) {
  gsub(" NA", "", x)
}))
colnames(I_table_pvalues) <- c("Algorithm", "AUC", "Brier", "PG", "EMP")
I_table_pvalues_latex <- format_p_values(kable(I_table_pvalues, "latex", booktabs = T))


#######################################################
# Bayesian signed rank test (Benavoli et al., 2017)
#######################################################
AUC_bayes <- perf_mod(AUC_prep_rank %>% select("id", "SRE_boosting", "SRE_RF", "LRR", "LGBM", "RF", "GAM", "CTREE", "PLTR"),
                      iter = 20000,
                      seed = 42)




AUC_SRE_LRR <- contrast_models(AUC_bayes, c(rep('SRE_boosting',1)), c("LRR"))
autoplot(AUC_SRE_LRR, size = 0.01, color = "darkgrey", linewidth=1) + 
  ggtitle("") + 
  xlab("") +
  ylab("") +
  theme_minimal() 
kable(summary(AUC_SRE_LRR, size = 0.01) %>% 
        dplyr::select(contrast, starts_with("pract")), "latex", booktabs = T)
#RF
AUC_SRE_RF <- contrast_models(AUC_bayes, c(rep('SRE_boosting',1)), c("RF"))
autoplot(AUC_SRE_RF, size = 0.01, color = "darkgrey", linewidth=1) + 
  ggtitle("") + 
  xlab("") +
  ylab("") +
  theme_minimal() 
kable(summary(AUC_SRE_RF, size = 0.01) %>% 
        dplyr::select(contrast, starts_with("pract")), "latex", booktabs = T)


#all
AUC_contrasts <- contrast_models(AUC_bayes)
autoplot(AUC_contrasts, size = 0.01)
kable(summary(AUC_contrasts, size = 0.01) %>% 
  dplyr::select(contrast, starts_with("pract")), "latex", booktabs = T)





#Scale between 0.5 and 1 NIET NODIG
#max_Brier <- Brier_prep_rank %>% dplyr::select(where(is.numeric)) %>% apply(1,max)
#Brier_scaled <- cbind(Brier_prep_rank[1],
#                     0.5 + 0.5*(Brier_prep_rank[-1]/max_Brier))
Brier_bayes <- perf_mod(Brier_prep_rank %>% select("id", "SRE_boosting", "SRE_RF", "LRR", "LGBM", "RF"), #NORMALISEREN
                      iter = 20000,
                      seed = 42)

Brier_SRE_LRR <- contrast_models(Brier_bayes, c(rep('SRE_boosting',1)), c("LRR"))
autoplot(Brier_SRE_LRR, size = 0.0025, color = "dodgerblue", linewidth=1) + 
  ggtitle("") + 
  xlab("") +
  ylab("") +
  theme_minimal() 
kable(summary(Brier_SRE_LRR, size = 0.0025) %>% 
        dplyr::select(contrast, starts_with("pract")), "latex", booktabs = T)

#RF
Brier_SRE_RF <- contrast_models(Brier_bayes, c(rep('SRE_boosting',1)), c("LGBM"))
autoplot(Brier_SRE_RF, size = 0.0025, color = "darkgrey", linewidth=1) + 
  ggtitle("") + 
  xlab("") +
  ylab("") +
  theme_minimal()
kable(summary(Brier_SRE_RF, size = 0.0025) %>% 
        dplyr::select(contrast, starts_with("pract")), "latex", booktabs = T)


Brier_SRE_LRR <- contrast_models(Brier_bayes, c(rep('SRE_boosting',2)), c("LRR", "LGBM"))
autoplot(Brier_SRE_LRR, size = 0.0025) + 
  ggtitle("Posterior distribution of differences: Brier Score") + 
  xlab("Difference in Brier Score (SRE_boosting - alternative)")
summary(Brier_SRE_LRR, size = 0.0025) %>% 
  dplyr::select(contrast, starts_with("pract"))
Brier_SRE_LRR <- contrast_models(Brier_bayes, 'SRE', 'LRR')
autoplot(Brier_SRE_LRR)
summary(Brier_SRE_LRR, size = 0.01) %>% 
  dplyr::select(contrast, starts_with("pract"))
Brier_RF_LRR <- contrast_models(Brier_bayes, 'RF', 'LRR')
autoplot(Brier_RF_LRR)
summary(Brier_RF_LRR, size = 0.01) %>% 
  dplyr::select(contrast, starts_with("pract"))

#all
Brier_contrasts <- contrast_models(Brier_bayes)
autoplot(Brier_contrasts, size = 0.01)
kable(summary(Brier_contrasts, size = 0.01) %>% 
  dplyr::select(contrast, starts_with("pract")), "latex", booktabs = T)


#Scale between 0.5 and 1
#max_PG <- PG_prep_rank %>% dplyr::select(where(is.numeric)) %>% apply(1,max)
#PG_scaled <- cbind(PG_prep_rank[1],
#                   0.5 + 0.5*(PG_prep_rank[-1]/max_PG))
PG_bayes <- perf_mod(PG_prep_rank %>% select("id", "SRE_boosting", "SRE_RF", "LRR", "LGBM", "RF"),
                     iter = 20000,
                     seed = 42)

#LRR
PG_SRE_LRR <- contrast_models(PG_bayes, c(rep('SRE_RF',1)), c("LRR"))
autoplot(PG_SRE_LRR, size = 0.01, color = "orangered", linewidth=1) + 
  ggtitle("") + 
  xlab("") +
  ylab("") +
  theme_minimal() 
kable(summary(PG_SRE_LRR, size = 0.01) %>% 
        dplyr::select(contrast, starts_with("pract")), "latex", booktabs = T)

#RF
PG_SRE_RF <- contrast_models(PG_bayes, c(rep('SRE_RF',1)), c("LGBM"))
autoplot(PG_SRE_RF, size = 0.01, color = "dodgerblue", linewidth=1) + 
  ggtitle("") + 
  xlab("") +
  ylab("") +
  theme_minimal()
kable(summary(PG_SRE_RF, size = 0.01) %>% 
        dplyr::select(contrast, starts_with("pract")), "latex", booktabs = T)


  
  
summary(PG_SRE_LRR, size = 0.01) %>% 
        dplyr::select(contrast, starts_with("pract"))


# EMP
#Scale between 0.5 and 1
#max_EMP <- EMP_prep_rank %>% dplyr::select(where(is.numeric)) %>% apply(1,max)
#EMP_scaled <- cbind(EMP_prep_rank[1],
#                   0.5 + 0.5*(EMP_prep_rank[-1]/max_EMP))
EMP_bayes <- perf_mod(EMP_prep_rank %>% select("id", "SRE_boosting", "SRE_RF", "LRR", "LGBM", "RF"),
                     iter = 20000,
                     seed = 42)

EMP_SRE_LRR <- contrast_models(EMP_bayes, c(rep('SRE_boosting',1)), c("LRR"))
autoplot(EMP_SRE_LRR, size = 0.001, color = "dodgerblue", linewidth=1) + 
  ggtitle("") + 
  xlab("") +
  ylab("") +
  theme_minimal() 
kable(summary(EMP_SRE_LRR, size = 0.001) %>% 
        dplyr::select(contrast, starts_with("pract")), "latex", booktabs = T)

#RF
EMP_SRE_RF <- contrast_models(EMP_bayes, c(rep('SRE_boosting',1)), c("RF"))
autoplot(EMP_SRE_RF, size = 0.001, color = "darkgrey", linewidth=1) + 
  ggtitle("") + 
  xlab("") +
  ylab("") +
  theme_minimal()
kable(summary(EMP_SRE_RF, size = 0.001) %>% 
        dplyr::select(contrast, starts_with("pract")), "latex", booktabs = T)


EMP_SRE_LRR <- contrast_models(EMP_bayes, c(rep('SRE_boosting',2)), c("LRR", "RF"))
autoplot(EMP_SRE_LRR, size = 0.001) + 
  ggtitle("Posterior distribution of differences: Expected Maximum Profit") + 
  xlab("Difference in EMP (SRE_boosting - alternative)")

summary(EMP_SRE_LRR, size = 0.001) %>% 
        dplyr::select(contrast, starts_with("pract"))



#Compare important algorithms
control <- c(rep('SRE_boosting',3), rep('SRE_bag',3))
compare <- rep(c("LRR", "RF", "SRE_boosting"),2)

control_small <- rep('SRE',2)
compare_small <- c("LRR", "RF")


AUC_comparison <- contrast_models(AUC_bayes, 
                                  control,
                                  compare)
plots <- autoplot(AUC_comparison, size = 0.01) +
  facet_wrap(~contrast, scales = "free", nrow = 2)
print(plots)

kable(summary(AUC_comparison, size = 0.01) %>% 
        dplyr::select(contrast, starts_with("pract")) %>%
        mutate_if(is.numeric, round, digits = 3), "latex", booktabs = T)


Brier_comparison <- contrast_models(Brier_bayes, 
                                    control,
                                    compare)
#kable(comparison_Brier, "latex", booktabs = T)
plots <- autoplot(Brier_comparison, size = 0.01) +
  facet_wrap(~contrast, scales = "free", nrow = 2)
print(plots)

kable(summary(Brier_comparison, size = 0.01) %>% 
        dplyr::select(contrast, starts_with("pract")) %>%
        mutate_if(is.numeric, round, digits = 3), "latex", booktabs = T)



PG_comparison <- contrast_models(PG_bayes, 
                                 control,
                                 compare)

plots <- autoplot(PG_comparison, size = 0.01) +
  facet_wrap(~contrast, scales = "free", nrow = 2)
print(plots)

kable(summary(PG_comparison, size = 0.01) %>% 
        dplyr::select(contrast, starts_with("pract")) %>%
        mutate_if(is.numeric, round, digits = 3), "latex", booktabs = T)



EMP_comparison <- contrast_models(EMP_bayes, 
                                 control,
                                 compare)

plots <- autoplot(EMP_comparison, size = 0.01) +
  facet_wrap(~contrast, scales = "free", nrow = 2)
print(plots)

kable(summary(EMP_comparison, size = 0.01) %>% 
        dplyr::select(contrast, starts_with("pract")) %>%
        mutate_if(is.numeric, round, digits = 3), "latex", booktabs = T)







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
  dplyr::select(-c(3))
names(AUC_results) <- c("algorithm", "AC", "TH02")
AUC_ranks <- data.frame(apply(AUC_results%>%dplyr::select(-1), MARGIN = 2, FUN = rank))%>%
  rename(rank_AC = AC) %>%
  rename(rank_TH02 = TH02)
avg_AUC_rank <- rowMeans(AUC_ranks)
AUC_results$avgRank_AUC <- avg_AUC_rank

BRIER_results <- do.call(cbind, avg_results[grepl("BRIER", names(avg_results))]) %>%
  dplyr::select(-c(3))
names(BRIER_results) <- c("algorithm", "AC", "TH02")
BRIER_ranks <- data.frame(apply(-(BRIER_results%>%dplyr::select(-1)), MARGIN = 2, FUN = rank))%>%
  rename(rank_AC = AC) %>%
  rename(rank_TH02 = TH02)
avg_BRIER_rank <- rowMeans(BRIER_ranks)
BRIER_results$avgRank_BRIER <- avg_BRIER_rank

PG_results <- do.call(cbind, avg_results[grepl("PG", names(avg_results))]) %>%
  dplyr::select(-c(3))
names(PG_results) <- c("algorithm", "AC", "TH02")
PG_ranks <- data.frame(apply(PG_results%>%dplyr::select(-1), MARGIN = 2, FUN = rank))%>%
  rename(rank_AC = AC) %>%
  rename(rank_TH02 = TH02)
avg_PG_rank <- rowMeans(PG_ranks)
PG_results$avgRank_PG <- avg_PG_rank

AUC_results




#regressions
dataset_sizes <- sapply(datasets, nrow)
datasets_numeric_cols = c()
datasets_nominal_cols = c()
datasets_proportion_nominal_cols = c()
datasets_prior = c()
datasets_cor = c()
for(i in 1:length(datasets)) {
  datasets_numeric_cols[i]<-sum(sapply(datasets[[i]], function(col) (!is.factor(col)&&!is.character(col))))
  
}

for(i in 1:length(datasets)) {
  column_counter = 0
  for(j in 1:ncol(datasets[[i]])) {
    if((is.factor(datasets[[i]][[j]])||is.character(datasets[[i]][[j]]))&&(colnames(datasets[[i]][j])!="label")) {
      column_counter = column_counter +1
    }
  }
  datasets_nominal_cols[i] <- column_counter
  datasets_proportion_nominal_cols[i] <- column_counter/ncol(datasets[[i]]%>%select(-label))
}

for(i in 1:length(datasets)) {
  prior <- (table(datasets[[i]]$label)["X1"])[[1]]/dataset_sizes[i]
  datasets_prior[i] <- prior
}

for(i in 1:length(datasets)) {
  dummy_creator <- recipe(label ~., datasets[[i]]) %>% step_naomit(all_predictors(), skip = F) %>% step_dummy(all_nominal_predictors()) %>% step_nzv(all_predictors())%>% prep()
  cordata <- dummy_creator %>% bake(datasets[[i]])
  abscor <- corr_abs(cordata%>%select(-label), cordata$label )
  datasets_cor[i] <- abscor
}


combined_results_AUC$size <- rep(0, nrow(combined_results_AUC))
combined_results_Brier$size <- rep(0, nrow(combined_results_AUC))
combined_results_PG$size <- rep(0, nrow(combined_results_AUC))
combined_results_AUC$numeric_cols <- rep(0, nrow(combined_results_AUC))
combined_results_Brier$numeric_cols <- rep(0, nrow(combined_results_AUC))
combined_results_PG$numeric_cols <- rep(0, nrow(combined_results_AUC))
combined_results_AUC$nominal_cols <- rep(0, nrow(combined_results_AUC))
combined_results_Brier$nominal_cols <- rep(0, nrow(combined_results_AUC))
combined_results_PG$nominal_cols <- rep(0, nrow(combined_results_AUC))
combined_results_AUC$proportion_nominal_cols <- rep(0, nrow(combined_results_AUC))
combined_results_Brier$proportion_nominal_cols <- rep(0, nrow(combined_results_AUC))
combined_results_PG$proportion_nominal_cols <- rep(0, nrow(combined_results_AUC))
combined_results_AUC$nr_cols <- rep(0, nrow(combined_results_AUC))
combined_results_Brier$nr_cols <- rep(0, nrow(combined_results_AUC))
combined_results_PG$nr_cols <- rep(0, nrow(combined_results_AUC))
combined_results_AUC$prior <- rep(0, nrow(combined_results_AUC))
combined_results_Brier$prior <- rep(0, nrow(combined_results_AUC))
combined_results_PG$prior <- rep(0, nrow(combined_results_AUC))
combined_results_AUC$cor <- rep(0, nrow(combined_results_AUC))
combined_results_Brier$cor <- rep(0, nrow(combined_results_AUC))
combined_results_PG$cor <- rep(0, nrow(combined_results_AUC))

for(i in 1:length(dataset_sizes)) {
  combined_results_AUC$size[((i-1)*90+1):(i*90)] <- rep(dataset_sizes[i], 90)
  combined_results_Brier$size[((i-1)*90+1):(i*90)] <- rep(dataset_sizes[i], 90)
  combined_results_PG$size[((i-1)*90+1):(i*90)] <- rep(dataset_sizes[i], 90)
  
  combined_results_AUC$numeric_cols[((i-1)*90+1):(i*90)] <- rep(datasets_numeric_cols[i], 90)
  combined_results_Brier$numeric_cols[((i-1)*90+1):(i*90)] <- rep(datasets_numeric_cols[i], 90)
  combined_results_PG$numeric_cols[((i-1)*90+1):(i*90)] <- rep(datasets_numeric_cols[i], 90)
  
  combined_results_AUC$nominal_cols[((i-1)*90+1):(i*90)] <- rep(datasets_nominal_cols[i], 90)
  combined_results_Brier$nominal_cols[((i-1)*90+1):(i*90)] <- rep(datasets_nominal_cols[i], 90)
  combined_results_PG$nominal_cols[((i-1)*90+1):(i*90)] <- rep(datasets_nominal_cols[i], 90)

  combined_results_AUC$proportion_nominal_cols[((i-1)*90+1):(i*90)] <- rep(datasets_proportion_nominal_cols[i], 90)
  combined_results_Brier$proportion_nominal_cols[((i-1)*90+1):(i*90)] <- rep(datasets_proportion_nominal_cols[i], 90)
  combined_results_PG$proportion_nominal_cols[((i-1)*90+1):(i*90)] <- rep(datasets_proportion_nominal_cols[i], 90)
  
  combined_results_AUC$nr_cols[((i-1)*90+1):(i*90)] <- rep(datasets_nominal_cols[i] + datasets_numeric_cols[i], 90)
  combined_results_Brier$nr_cols[((i-1)*90+1):(i*90)] <- rep(datasets_nominal_cols[i] + datasets_numeric_cols[i], 90)
  combined_results_PG$nr_cols[((i-1)*90+1):(i*90)] <- rep(datasets_nominal_cols[i] + datasets_numeric_cols[i], 90)
  
  combined_results_AUC$prior[((i-1)*90+1):(i*90)] <- rep(datasets_prior[i], 90)
  combined_results_Brier$prior[((i-1)*90+1):(i*90)] <- rep(datasets_prior[i], 90)
  combined_results_PG$prior[((i-1)*90+1):(i*90)] <- rep(datasets_prior[i], 90)
  
  combined_results_AUC$cor[((i-1)*90+1):(i*90)] <- rep(datasets_cor[i], 90)
  combined_results_Brier$cor[((i-1)*90+1):(i*90)] <- rep(datasets_cor[i], 90)
  combined_results_PG$cor[((i-1)*90+1):(i*90)] <- rep(datasets_cor[i], 90)
}

# Compare SRE and LRR
comparison_AUC <- combined_results_AUC %>%
  filter(algorithm %in% c("LRR", "SRE_boosting")) %>%
  spread(key = algorithm, value = metric) %>%
  mutate(SRE_better_than_LRR = LRR < SRE_boosting)
comparison_Brier <- combined_results_Brier %>%
  filter(algorithm %in% c("LRR", "SRE_boosting")) %>%
  spread(key = algorithm, value = metric) %>%
  mutate(SRE_better_than_LRR = LRR > SRE_boosting)
comparison_PG <- combined_results_PG %>%
  filter(algorithm %in% c("LRR", "SRE_RF")) %>%
  spread(key = algorithm, value = metric) %>%
  mutate(SRE_better_than_LRR = LRR < SRE_RF)

basetable_AUC <- comparison_AUC %>%
  dplyr::select(c(size, numeric_cols, nominal_cols, proportion_nominal_cols, nr_cols, prior, SRE_better_than_LRR, cor)) %>%
  mutate(feature_ratio = nominal_cols/numeric_cols) %>%
  mutate(SRE_better_than_LRR = as.factor(SRE_better_than_LRR))
basetable_Brier <- comparison_Brier %>%
  dplyr::select(c(size, numeric_cols, nominal_cols, proportion_nominal_cols, nr_cols, prior, SRE_better_than_LRR, cor)) %>%
  mutate(feature_ratio = nominal_cols/numeric_cols) %>%
  mutate(SRE_better_than_LRR = as.factor(SRE_better_than_LRR))
basetable_PG <- comparison_PG %>%
  dplyr::select(c(size, numeric_cols, nominal_cols, proportion_nominal_cols, nr_cols, prior, SRE_better_than_LRR, cor)) %>%
  mutate(feature_ratio = nominal_cols/numeric_cols) %>%
  mutate(SRE_better_than_LRR = as.factor(SRE_better_than_LRR))


#CTREE

AUC_tree <- ctree(SRE_better_than_LRR ~., basetable_AUC, control = ctree_control(testtype = c("Bonferroni"), mincriterion = 0.9)) # 1 node
Brier_tree <- ctree(SRE_better_than_LRR ~., basetable_Brier, control = ctree_control(testtype = c("Bonferroni"), mincriterion = 0.9)) # 1 node
PG_tree <- ctree(SRE_better_than_LRR ~., basetable_PG, control = ctree_control(testtype = c("Bonferroni"), mincriterion = 0.9))

plot(AUC_tree, drop_terminal = F, type = "simple")
plot(Brier_tree, drop_terminal = F, type = "simple")
plot(PG_tree, drop_terminal = F, type = "simple")

#####
#table for attachments
combined_results_AUC_table <- combined_results_AUC %>%
  group_by(dataset, algorithm) %>%
  summarise(avg_metric = round(mean(metric), 3), sd_metric = round(sd(metric), 3)) %>%
  ungroup() %>%
  mutate_if(is.numeric, ~scales::number(., accuracy = 0.001))

combined_results_AUC_table$sd_brackets <- mapply(paste, "(", combined_results_AUC_table$sd_metric, ")", MoreArgs = list(sep = ""))
combined_results_AUC_table$AUC <- mapply(paste, combined_results_AUC_table$avg_metric, combined_results_AUC_table$sd_brackets, MoreArgs = list(sep = " "))

finished_AUC_table <- combined_results_AUC_table %>%
  dplyr::select(dataset, algorithm, AUC) %>%
  pivot_wider(names_from = dataset, values_from = AUC)

combined_results_Brier_table <- combined_results_Brier %>%
  group_by(dataset, algorithm) %>%
  summarise(avg_metric = round(mean(metric), 3), sd_metric = round(sd(metric), 3)) %>%
  ungroup() %>%
  mutate_if(is.numeric, ~scales::number(., accuracy = 0.001))

combined_results_Brier_table$sd_brackets <- mapply(paste, "(", combined_results_Brier_table$sd_metric, ")", MoreArgs = list(sep = ""))
combined_results_Brier_table$Brier <- mapply(paste, combined_results_Brier_table$avg_metric, combined_results_Brier_table$sd_brackets, MoreArgs = list(sep = " "))

finished_Brier_table <- combined_results_Brier_table %>%
  dplyr::select(dataset, algorithm, Brier) %>%
  pivot_wider(names_from = dataset, values_from = Brier)

combined_results_PG_table <- combined_results_PG %>%
  group_by(dataset, algorithm) %>%
  summarise(avg_metric = round(mean(metric), 3), sd_metric = round(sd(metric), 3)) %>%
  ungroup() %>%
  mutate_if(is.numeric, ~scales::number(., accuracy = 0.001))

combined_results_PG_table$sd_brackets <- mapply(paste, "(", combined_results_PG_table$sd_metric, ")", MoreArgs = list(sep = ""))
combined_results_PG_table$PG <- mapply(paste, combined_results_PG_table$avg_metric, combined_results_PG_table$sd_brackets, MoreArgs = list(sep = " "))

finished_PG_table <- combined_results_PG_table %>%
  dplyr::select(dataset, algorithm, PG) %>%
  pivot_wider(names_from = dataset, values_from = PG)

kable(rbind(finished_AUC_table, finished_Brier_table, finished_PG_table), "latex", booktabs = T)



# Comparison with configuration of De Bock
combined_results_AUC_DB_config$algorithm <- paste(combined_results_AUC_DB_config$algorithm, "_AP", sep = "")
combined_results_Brier_DB_config$algorithm <- paste(combined_results_Brier_DB_config$algorithm, "_AP", sep = "")
combined_results_PG_DB_config$algorithm <- paste(combined_results_PG_DB_config$algorithm, "_AP", sep = "")

DB_basetable_AUC <- rbind(combined_results_AUC %>% dplyr::filter(algorithm=="SRE"|algorithm=="RF"), combined_results_AUC_DB_config%>%filter(algorithm!="RE_AP"))
DB_basetable_Brier <- rbind(combined_results_Brier %>% dplyr::filter(algorithm=="SRE"|algorithm=="RF"), combined_results_Brier_DB_config%>%filter(algorithm!="RE_AP"))
DB_basetable_PG <- rbind(combined_results_PG %>% dplyr::filter(algorithm=="SRE"|algorithm=="RF"), combined_results_PG_DB_config%>%filter(algorithm!="RE_AP"))

#for latex table 
DB_basetable_AUC_summarized <- DB_basetable_AUC %>%
  group_by(algorithm) %>%
  summarise("Avg" = round(mean(metric), 3), "stdev" = round(sd(metric),3)) %>%
  ungroup() %>%
  mutate_if(is.numeric, ~scales::number(., accuracy = 0.001))

DB_basetable_AUC_summarized$sd_brackets <- mapply(paste, "(", DB_basetable_AUC_summarized$stdev, ")", MoreArgs = list(sep = ""))
DB_basetable_AUC_summarized$AUC <- mapply(paste, DB_basetable_AUC_summarized$Avg, DB_basetable_AUC_summarized$sd_brackets, MoreArgs = list(sep = " "))

finished_AUC_DB_table_summarized <- DB_basetable_AUC_summarized %>%
  dplyr::select(algorithm, AUC) %>%
  pivot_wider(names_from = algorithm, values_from = AUC)

DB_basetable_Brier_summarized <- DB_basetable_Brier %>%
  group_by(algorithm) %>%
  summarise("Avg" = round(mean(metric), 3), "stdev" = round(sd(metric),3)) %>%
  ungroup() %>%
  mutate_if(is.numeric, ~scales::number(., accuracy = 0.001))

DB_basetable_Brier_summarized$sd_brackets <- mapply(paste, "(", DB_basetable_Brier_summarized$stdev, ")", MoreArgs = list(sep = ""))
DB_basetable_Brier_summarized$Brier <- mapply(paste, DB_basetable_Brier_summarized$Avg, DB_basetable_Brier_summarized$sd_brackets, MoreArgs = list(sep = " "))

finished_Brier_DB_table_summarized <- DB_basetable_Brier_summarized %>%
  dplyr::select(algorithm, Brier) %>%
  pivot_wider(names_from = algorithm, values_from = Brier)

DB_basetable_PG_summarized <- DB_basetable_PG %>%
  group_by(algorithm) %>%
  summarise("Avg" = round(mean(metric), 3), "stdev" = round(sd(metric),3)) %>%
  ungroup() %>%
  mutate_if(is.numeric, ~scales::number(., accuracy = 0.001))

DB_basetable_PG_summarized$sd_brackets <- mapply(paste, "(", DB_basetable_PG_summarized$stdev, ")", MoreArgs = list(sep = ""))
DB_basetable_PG_summarized$PG <- mapply(paste, DB_basetable_PG_summarized$Avg, DB_basetable_PG_summarized$sd_brackets, MoreArgs = list(sep = " "))

finished_PG_DB_table_summarized <- DB_basetable_PG_summarized %>%
  dplyr::select(algorithm, PG) %>%
  pivot_wider(names_from = algorithm, values_from = PG)


kable(rbind(finished_AUC_DB_table_summarized, finished_Brier_DB_table_summarized, finished_PG_DB_table_summarized), "latex", booktabs=T)
#####
average_ranks_AUC <- avg_ranks(DB_basetable_AUC)
average_ranks_Brier <- avg_ranks(DB_basetable_Brier, direction = "min")
average_ranks_PG <- avg_ranks(DB_basetable_PG)

avg_ranks_summarized_AUC <- avg_ranks_summarized(average_ranks_AUC)
avg_ranks_summarized_Brier <- avg_ranks_summarized(average_ranks_Brier)
avg_ranks_summarized_PG <- avg_ranks_summarized(average_ranks_PG)


#bayesian comparison
DB_basetable_AUC$group <- paste(DB_basetable_AUC$dataset, DB_basetable_AUC$nr_fold)
AUC_prep_rank <- DB_basetable_AUC %>% dplyr::select(group, algorithm, metric) %>% pivot_wider(names_from = algorithm, values_from = metric) %>% rename("id" = group)
names(AUC_prep_rank) <- c("id", "RF", "SRE", "RF_AP", "SRE_AP")
AUC_bayes <- perf_mod(AUC_prep_rank,
                      iter = 40000,
                      seed = 42,
                      chains = 4)

#all
AUC_contrasts <- contrast_models(AUC_bayes)
autoplot(AUC_contrasts, size = 0.01)
latex_summary_AUC_DB <-summary(AUC_contrasts, size = 0.01) %>% 
        dplyr::select(contrast, starts_with("pract"))





DB_basetable_Brier$group <- paste(DB_basetable_Brier$dataset, DB_basetable_Brier$nr_fold)
Brier_prep_rank <- DB_basetable_Brier %>% dplyr::select(group, algorithm, metric) %>% pivot_wider(names_from = algorithm, values_from = metric) %>% rename("id" = group)
#Scale between 0.5 and 1
max_Brier <- Brier_prep_rank %>% dplyr::select(where(is.numeric)) %>% apply(1,max)
Brier_scaled <- cbind(Brier_prep_rank[1],
                      0.5 + 0.5*(Brier_prep_rank[-1]/max_Brier))
names(Brier_scaled) <- c("id", "RF", "SRE", "RF_AP", "SRE_AP")
Brier_bayes <- perf_mod(Brier_scaled, #NORMALISEREN
                        iter = 40000,
                        seed = 42)

#all
Brier_contrasts <- contrast_models(Brier_bayes)
autoplot(Brier_contrasts, size = 0.01)
latex_summary_Brier_DB <- summary(Brier_contrasts, size = 0.01) %>% 
        dplyr::select(contrast, starts_with("pract"))


DB_basetable_PG$group <- paste(DB_basetable_PG$dataset, DB_basetable_PG$nr_fold)
PG_prep_rank <- DB_basetable_PG %>% dplyr::select(group, algorithm, metric) %>% pivot_wider(names_from = algorithm, values_from = metric) %>% rename("id" = group)
#Scale between 0.5 and 1
max_PG <- PG_prep_rank %>% dplyr::select(where(is.numeric)) %>% apply(1,max)
PG_scaled <- cbind(PG_prep_rank[1],
                   0.5 + 0.5*(PG_prep_rank[-1]/max_PG))
names(PG_scaled) <- c("id", "RF", "SRE", "RF_AP", "SRE_AP")
PG_bayes <- perf_mod(PG_scaled,
                     iter = 40000,
                     seed = 42)

#all
PG_contrasts <- contrast_models(PG_bayes)
autoplot(PG_contrasts, size = 0.01)
latex_summary_PG_DB <- summary(PG_contrasts, size = 0.01) %>% 
        dplyr::select(contrast, starts_with("pract"))

kable(rbind(latex_summary_AUC_DB, latex_summary_Brier_DB, latex_summary_PG_DB), "latex", booktabs = T)

#####
#table for attachments?
combined_results_AUC_DB_table <- DB_basetable_AUC %>%
  group_by(dataset, algorithm) %>%
  summarise(avg_metric = round(mean(metric), 3), sd_metric = round(sd(metric), 3)) %>%
  ungroup() %>%
  mutate_if(is.numeric, ~scales::number(., accuracy = 0.001))

combined_results_AUC_DB_table$sd_brackets <- mapply(paste, "(", combined_results_AUC_DB_table$sd_metric, ")", MoreArgs = list(sep = ""))
combined_results_AUC_DB_table$AUC <- mapply(paste, combined_results_AUC_DB_table$avg_metric, combined_results_AUC_DB_table$sd_brackets, MoreArgs = list(sep = " "))

finished_AUC_DB_table <- combined_results_AUC_DB_table %>%
  dplyr::select(dataset, algorithm, AUC) %>%
  pivot_wider(names_from = dataset, values_from = AUC)

combined_results_Brier_DB_table <- DB_basetable_Brier %>%
  group_by(dataset, algorithm) %>%
  summarise(avg_metric = round(mean(metric), 3), sd_metric = round(sd(metric), 3)) %>%
  ungroup() %>%
  mutate_if(is.numeric, ~scales::number(., accuracy = 0.001))

combined_results_Brier_DB_table$sd_brackets <- mapply(paste, "(", combined_results_Brier_DB_table$sd_metric, ")", MoreArgs = list(sep = ""))
combined_results_Brier_DB_table$Brier <- mapply(paste, combined_results_Brier_DB_table$avg_metric, combined_results_Brier_DB_table$sd_brackets, MoreArgs = list(sep = " "))

finished_Brier_DB_table <- combined_results_Brier_DB_table %>%
  dplyr::select(dataset, algorithm, Brier) %>%
  pivot_wider(names_from = dataset, values_from = Brier)

combined_results_PG_DB_table <- DB_basetable_PG %>%
  group_by(dataset, algorithm) %>%
  summarise(avg_metric = round(mean(metric), 3), sd_metric = round(sd(metric), 3)) %>%
  ungroup() %>%
  mutate_if(is.numeric, ~scales::number(., accuracy = 0.001))

combined_results_PG_DB_table$sd_brackets <- mapply(paste, "(", combined_results_PG_DB_table$sd_metric, ")", MoreArgs = list(sep = ""))
combined_results_PG_DB_table$PG <- mapply(paste, combined_results_PG_DB_table$avg_metric, combined_results_PG_DB_table$sd_brackets, MoreArgs = list(sep = " "))

finished_PG_DB_table <- combined_results_PG_DB_table %>%
  dplyr::select(dataset, algorithm, PG) %>%
  pivot_wider(names_from = dataset, values_from = PG)

kable(rbind(finished_AUC_DB_table, finished_Brier_DB_table, finished_PG_DB_table), "latex", booktabs = T)
