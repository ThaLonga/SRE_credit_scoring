if (!require("pacman")) install.packages("pacman") ; require("pacman")
p_load(tidyverse, fastDummies, dplyr, DescTools, readxl)

source('./src/preprocessing_functions.R')





#German
german <- read_table("data/statlog+german+credit+data/german.csv", col_names = FALSE)

## change 1,2 to 0,1
german$X21 <- as.factor(ifelse(german$X21==1, 0, 1))
german <- german  %>% rename("label" = X21) %>%
  mutate(label = factor(label, 
                        labels = make.names(levels(label))))

save(german, file = "C:/Users/simon/Documents/GitHub/Thesis/data/GOLD/german.Rda")


#Australian
australian <- read_table("data/statlog+australian+credit+approval/australian.dat", col_names = FALSE)

australian <- australian %>%
  mutate_at(vars(X1, X4, X5, X6, X8, X9, X11, X12, X15), as.factor) %>%
  rename("label" = "X15") %>%
  mutate(label = factor(label, 
                        labels = make.names(levels(label))))

save(australian, file = "C:/Users/simon/Documents/GitHub/Thesis/data/GOLD/australian.Rda")

#HMEQ
HMEQ <- read_csv("data/HMEQ/hmeq.csv")

HMEQ <- HMEQ %>%
  mutate_at(vars(BAD), as.factor) %>%
  rename("label" = "BAD") %>%
  mutate(label = factor(label, 
                        labels = make.names(levels(label))))

HMEQ <- remove_rows_with_nas(HMEQ) #removes rows with >= 6 NAs

#add flags for missing values
columns_to_flag <- colnames(HMEQ)[unlist(lapply(HMEQ, function(x) sum(is.na(x))))>0]
for (col in columns_to_flag) {
  flag_col_name <- paste0(col, "_FLAG")
  HMEQ[[flag_col_name]] <- as.factor(ifelse(is.na(HMEQ[[col]]), 1, 0))
}

save(HMEQ, file = "C:/Users/simon/Documents/GitHub/Thesis/data/GOLD/hmeq.Rda")



#GMSC
kaggle <- read_csv("data/GiveMeSomeCredit/cs-training.csv")

# take a 20% sample
set.seed(42)
kaggle <- kaggle[sample(nrow(kaggle), round(0.2*nrow(kaggle))), ]

kaggle <- kaggle %>%
  mutate_at(vars(SeriousDlqin2yrs), as.factor) %>%
  rename("label" = "SeriousDlqin2yrs") %>%
  rename("NumberOfTime30To59DaysPastDueNotWorse" = "NumberOfTime30-59DaysPastDueNotWorse") %>%
  rename("NumberOfTime60To89DaysPastDueNotWorse" = "NumberOfTime60-89DaysPastDueNotWorse") %>%
  mutate(label = factor(label, 
                        labels = make.names(levels(label)))) %>%
  select(-...1)

#add flags for missing values
kaggle$MonthlyIncome_flag <- as.factor(ifelse(((kaggle$MonthlyIncome == 0)|is.na(kaggle$MonthlyIncome)), 1, 0))
kaggle$NumberOfDependents_flag <- as.factor(ifelse(is.na(kaggle$NumberOfDependents), 1, 0))

save(kaggle, file = "C:/Users/simon/Documents/GitHub/Thesis/data/GOLD/kaggle.Rda")


#TH02
thomas <- read_delim("data/02thomas/Loan_Data.csv", delim = ";", escape_double = FALSE, trim_ws = TRUE)

thomas <- thomas %>%
  mutate_at(vars(BAD, PHON), as.factor) %>%
  rename("label" = "BAD") %>%
  mutate(label = factor(label, 
                        labels = make.names(levels(label))))

thomas$YOB_flag <- as.factor(ifelse(thomas$YOB == 99, 1, 0))
thomas$DHVAL_flag <- as.factor(ifelse(thomas$DHVAL == 0, 1, 0))
thomas$DMORT_flag <- as.factor(ifelse(thomas$DMORT == 0, 1, 0))

save(thomas, file = "C:/Users/simon/Documents/GitHub/Thesis/data/GOLD/thomas.Rda")

#LC
LC <- read_delim("data/LC/LC.csv", delim = ",", escape_double = FALSE, trim_ws = TRUE, col_names = FALSE)

LC <- LC %>%
  mutate_at(vars(X2, X11), as.factor) %>%
  rename("label" = "X11") %>%
  mutate(label = factor(label, 
                        labels = make.names(levels(label))))

save(LC, file = "C:/Users/simon/Documents/GitHub/Thesis/data/GOLD/LC.Rda")

#JC
TC <- read_excel("data/default+of+credit+card+clients/default of credit card clients.xls", trim_ws = TRUE, range=cell_rows(2:30002), col_names = TRUE)
#33% sample
TC <- TC[sample(nrow(TC), round(1/3*nrow(TC))), ]
TC <- TC %>%
  mutate_all(as.integer) %>%
  rename("label" = "default payment next month") %>%
  mutate_at(vars(label), as.factor) %>%
  mutate(label = factor(label, 
                        labels = make.names(levels(label))))

save(TC, file = "C:/Users/simon/Documents/GitHub/Thesis/data/GOLD/TC.Rda")