library(pacman)
p_load(tidyverse, knitr, rstatix, tidyposterior, ggplot2, partykit, xtable)
source("./src/data_loader.R")
source("./src/results_processing_functions.R")
source("./src/adjust_Rom.R")
loaded_results <- load_results()
loaded_results_PLTR <- load_results("results_supp")
loaded_results_DB <- load_results("results_DB")
datasets <- load_data()
nr_datasets = 8


combined_results_AUC <- loaded_results[names(loaded_results) %>% grep("RF_AUC", .)] %>% 
  bind_rows() %>%
  select(-...1)  # Adjust the column name as necessary
combined_results_Brier <- loaded_results[names(loaded_results) %>% grep("RF_BRIER", .)] %>% 
  bind_rows() %>%
  select(-...1)  # Adjust the column name as necessary
combined_results_PG <- loaded_results[names(loaded_results) %>% grep("RF_PG", .)] %>% 
  bind_rows() %>%
  select(-...1)  # Adjust the column name as necessary
combined_results_EMP <- loaded_results[names(loaded_results) %>% grep("RF_EMP", .)] %>% 
  bind_rows() %>%
  select(-...1)  # Adjust the column name as necessary

#For no duplicate code
#combined_results_AUC <- combined_results_AUC_DB_config
#combined_results_Brier <- combined_results_Brier_DB_config
#combined_results_PG <- combined_results_PG_DB_config

# tables for attachments:
#####
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
#####


# AvgRank calculation

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

View(avg_ranks_summarized_AUC)
View(average_ranks_AUC)

# Friedman test
friedman_AUC <- average_ranks_AUC %>% convert_as_factor(dataset, algorithm) %>% dplyr::select(-average_metric) %>% friedman_test(average_rank ~ algorithm|dataset)
friedman_Brier <- average_ranks_Brier %>% convert_as_factor(dataset, algorithm) %>% friedman_test(average_rank ~ algorithm|dataset)
friedman_PG <- average_ranks_PG %>% convert_as_factor(dataset, algorithm) %>% friedman_test(average_rank ~ algorithm|dataset)


#AUC pairwise friedman
AUC_pairwise_p_values <- c()
for(i in 1:nrow(avg_ranks_summarized_AUC)) {
  R_j <- min(avg_ranks_summarized_AUC$average_rank)
  z <- friedman_pairwise(R_j, avg_ranks_summarized_AUC$average_rank[i], nrow(avg_ranks_summarized_AUC), nr_datasets)
  AUC_pairwise_p_values[i] <- pnorm(z, lower.tail = FALSE)*2
}
AUC_pairwise_p_values_adjusted <- adjustRom(AUC_pairwise_p_values, alpha=0.05)

#Brier pairwise friedman
Brier_pairwise_p_values <- c()
for(i in 1:nrow(avg_ranks_summarized_Brier)) {
  R_j <- min(avg_ranks_summarized_Brier$average_rank)
  z <- friedman_pairwise(R_j, avg_ranks_summarized_Brier$average_rank[i], nrow(avg_ranks_summarized_Brier), nr_datasets)
  Brier_pairwise_p_values[i] <- pnorm(z, lower.tail = FALSE)*2
}
Brier_pairwise_p_values_adjusted <- adjustRom(Brier_pairwise_p_values, alpha=0.05)

#PG pairwise friedman
PG_pairwise_p_values <- c()
for(i in 1:nrow(avg_ranks_summarized_PG)) {
  R_j <- min(avg_ranks_summarized_PG$average_rank)
  z <- friedman_pairwise(R_j, avg_ranks_summarized_PG$average_rank[i], nrow(avg_ranks_summarized_PG), nr_datasets)
  PG_pairwise_p_values[i] <- pnorm(z, lower.tail = FALSE)*2
}
PG_pairwise_p_values_adjusted <- adjustRom(PG_pairwise_p_values, alpha=0.05)

#join to make table
all_avg_ranks <- cbind(avg_ranks_summarized_AUC$algorithm, round(avg_ranks_summarized_AUC$average_rank, 2), round(avg_ranks_summarized_Brier$average_rank, 2), round(avg_ranks_summarized_PG$average_rank, 2)) %>% as_tibble()
pairwise_p_values <- cbind(avg_ranks_summarized_AUC$algorithm, round(AUC_pairwise_p_values_adjusted, 3), round(Brier_pairwise_p_values_adjusted, 3), round(PG_pairwise_p_values_adjusted, 3)) %>% as_tibble()
pairwise_p_values_brackets <- as.data.frame(mapply(paste, "(", pairwise_p_values, ")", MoreArgs = list(sep = "")))
pairwise_p_values_brackets[1]<-NA

table <- as.tibble(mapply(paste, all_avg_ranks, pairwise_p_values_brackets, MoreArgs = list(sep = " ")))
table <- as.tibble(lapply(table, function(x) {
  gsub(" NA", "", x)
}))
colnames(table) <- c("Algorithm", "AUC", "Brier", "PG")
table_latex <- kable(table, "latex", booktabs = T)

#####
# Bayesian signed rank test (Benavoli et al., 2017)
combined_results_AUC$group <- paste(combined_results_AUC$dataset, combined_results_AUC$nr_fold)
AUC_prep_rank <- combined_results_AUC %>% dplyr::select(group, algorithm, metric) %>% pivot_wider(names_from = algorithm, values_from = metric) %>% rename("id" = group)
AUC_bayes <- perf_mod(AUC_prep_rank,
                      iter = 50000,
                      seed = 42)

AUC_SRE_RE <- contrast_models(AUC_bayes, c(rep('SRE_boosting',4), rep('RE_boosting', 4)), rep(c('LRR', 'RF', 'RE_boosting', 'SRE_boosting'),2))
autoplot(AUC_SRE_RE, size = 0.01)
summary(AUC_SRE_RE, size = 0.01) %>% 
  dplyr::select(contrast, starts_with("pract"))
AUC_SRE_LRR <- contrast_models(AUC_bayes, 'SRE', 'LRR')
autoplot(AUC_SRE_LRR, size = 0.01)
summary(AUC_SRE_LRR, size = 0.01) %>% 
  dplyr::select(contrast, starts_with("pract"))
AUC_RF_LRR <- contrast_models(AUC_bayes, 'RF', 'LRR')
autoplot(AUC_RF_LRR, size = 0.01)
summary(AUC_RF_LRR, size = 0.01) %>% 
  dplyr::select(contrast, starts_with("pract"))

#all
AUC_contrasts <- contrast_models(AUC_bayes)
autoplot(AUC_contrasts, size = 0.01)
kable(summary(AUC_contrasts, size = 0.01) %>% 
  dplyr::select(contrast, starts_with("pract")), "latex", booktabs = T)





combined_results_Brier$group <- paste(combined_results_Brier$dataset, combined_results_Brier$nr_fold)
Brier_prep_rank <- combined_results_Brier %>% dplyr::select(group, algorithm, metric) %>% pivot_wider(names_from = algorithm, values_from = metric) %>% rename("id" = group)
#Scale between 0.5 and 1
max_Brier <- Brier_prep_rank %>% dplyr::select(where(is.numeric)) %>% apply(1,max)
Brier_scaled <- cbind(Brier_prep_rank[1],
                     0.5 + 0.5*(Brier_prep_rank[-1]/max_Brier))
Brier_bayes <- perf_mod(Brier_scaled, #NORMALISEREN
                      iter = 20000,
                      seed = 42)

Brier_SRE_RE <- contrast_models(Brier_bayes, 'SRE', 'RE')
autoplot(Brier_SRE_RE)
summary(Brier_SRE_RE, size = 0.01) %>% 
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


combined_results_PG$group <- paste(combined_results_PG$dataset, combined_results_PG$nr_fold)
PG_prep_rank <- combined_results_PG %>% dplyr::select(group, algorithm, metric) %>% pivot_wider(names_from = algorithm, values_from = metric) %>% rename("id" = group)
#Scale between 0.5 and 1
max_PG <- PG_prep_rank %>% dplyr::select(where(is.numeric)) %>% apply(1,max)
PG_scaled <- cbind(PG_prep_rank[1],
                   0.5 + 0.5*(PG_prep_rank[-1]/max_PG))
PG_bayes <- perf_mod(PG_scaled,
                     iter = 20000,
                     seed = 42)

PG_contrasts <- contrast_models(PG_bayes)
autoplot(PG_contrasts, size = 0.01)
kable(summary(PG_contrasts, size = 0.01) %>% 
        dplyr::select(contrast, starts_with("pract")),format = "latex", booktabs = T)


# EMP
combined_results_EMP$group <- paste(combined_results_EMP$dataset, combined_results_EMP$nr_fold)
EMP_prep_rank <- combined_results_EMP %>% dplyr::select(group, algorithm, metric) %>% pivot_wider(names_from = algorithm, values_from = metric) %>% rename("id" = group)
#Scale between 0.5 and 1
max_EMP <- EMP_prep_rank %>% dplyr::select(where(is.numeric)) %>% apply(1,max)
EMP_scaled <- cbind(EMP_prep_rank[1],
                   0.5 + 0.5*(EMP_prep_rank[-1]/max_EMP))
EMP_bayes <- perf_mod(EMP_scaled,
                     iter = 20000,
                     seed = 42)

EMP_contrasts <- contrast_models(EMP_bayes)
autoplot(EMP_contrasts, size = 0.01)
kable(summary(EMP_contrasts, size = 0.01) %>% 
        dplyr::select(contrast, starts_with("pract")),format = "latex", booktabs = T)



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
  filter(algorithm %in% c("LRR", "SRE")) %>%
  spread(key = algorithm, value = metric) %>%
  mutate(SRE_better_than_LRR = LRR < SRE)
comparison_Brier <- combined_results_Brier %>%
  filter(algorithm %in% c("LRR", "SRE")) %>%
  spread(key = algorithm, value = metric) %>%
  mutate(SRE_better_than_LRR = LRR > SRE)
comparison_PG <- combined_results_PG %>%
  filter(algorithm %in% c("LRR", "SRE")) %>%
  spread(key = algorithm, value = metric) %>%
  mutate(SRE_better_than_LRR = LRR < SRE)

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
