# Case study
#fold 1 on LC dataset (SRE_pg)
if (!require("pacman")) install.packages("pacman") ; require("pacman")
p_load(tidyverse, parsnip, tidymodels, stringr, knitr)
SRE <- readRDS("./casestudy/SRE_example_model.RDa")
rule_dictionary <- readRDS("./casestudy/rule_dictionary.RDa")
basetable <- readRDS("./casestudy/final_basetable_PG.RDa")
basetable_test <- readRDS("./casestudy/test_basetable.RDa")

SRE_tidy <- SRE %>% extract_fit_parsnip() %>% tidy() %>% filter(estimate !=0) 
rule_numbers <- as.numeric(lapply(SRE_tidy$term, function(x) str_replace_all(str_extract(x, "_(\\d+)"), "_", "")))
rule_numbers <- rule_numbers[!is.na(rule_numbers)]
rule_ids <- paste0("rule", rule_numbers)

selected_rules <- rule_dictionary[rule_ids, ]
SRE_tidy_rules <- SRE_tidy[2:11,] %>% cbind(selected_rules)  %>% dplyr::select(-penalty, -rule)


rule_importance <- c()
for(i in 1:nrow(SRE_tidy_rules)) {
  s_j <- sum(basetable[SRE_tidy_rules[i,"term"]][[1]])/nrow(basetable)
  rule_importance[i] <- abs(SRE_tidy_rules[i,"estimate"])*sqrt(s_j*(1-s_j))
}

linear_importance <- abs(SRE_tidy[12, "estimate"][[1]])*sd(basetable$winsorized_truncate_X3)
smooth_importance <- abs(SRE_tidy[13, "estimate"][[1]])*sd(basetable$`winsorized_truncate_s(X3)`)
importances <- c(rule_importance, linear_importance, smooth_importance)
norm_imp <- importances/max(importances)*100

SRE_rules_col <- c(SRE_tidy_rules$description,NA,NA)
importance_SRE <- cbind(SRE_tidy[-1,], norm_imp, SRE_rules_col) %>% arrange(desc(norm_imp)) %>% dplyr::select(term, SRE_rules_col, estimate, norm_imp)
names(importance_SRE) <- c("Term", "Rule conditions", "Coefficient", "Term importance")

kable(importance_SRE, "latex", booktabs = T)

# The prediction
basetable_test[which.max(y_pred),] %>% dplyr::select(all_of(importance_SRE$Term)) %>% kable('latex', booktabs=T)



#plotting
plot(final_GAM_fit%>%extract_fit_engine(), ylim = c(-2,2), xlim = c(-3,3))
plot(x = NA, y = NA, xlim = c(0, 3), ylim = c(-3, 1), xlab = "X3", ylab = "l(X3)", xaxs = "i", yaxs = "i")
abline(a=0, b=SRE_tidy[12, "estimate"][[1]], lwd=2)
