# Case study
if (!require("pacman")) install.packages("pacman") ; require("pacman")
p_load(tidyverse, parsnip, tidymodels, stringr)
SRE <- readRDS("./casestudy/SRE_example_model.RDa")
rule_dictionary <- readRDS("./casestudy/rules_dictionary.RDa")

SRE_tidy <- SRE %>% extract_fit_parsnip() %>% tidy() %>% filter(estimate !=0) 
rule_numbers <- as.numeric(lapply(SRE_tidy$term, function(x) str_replace_all(str_extract(x, "_(\\d+)"), "_", "")))[-1]
rule_dictionary[rule_numbers, ]
