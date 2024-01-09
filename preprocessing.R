if (!require("pacman")) install.packages("pacman") ; require("pacman")
p_load(tidyverse, fastDummies, dplyr, DescTools)

source('./src/preprocessing_functions.R')

#German
german <- read_table("data/statlog+german+credit+data/german.csv", col_names = FALSE)

#names(german) <- c("Status_existing_checking_account", "Duration_month", "Credit_history", "Purpose", "amount",
#                   "Savings_account_bonds", "Present_employment_since", "Installment_rate_percentage_disposable_income",
#                   "Personal_status_and_sex", "Other_debtors_guarantors", "Present_residence_since",
#                   "Property", "Age", "Other_installment_plans", "Housing", "existing_credits", "Job",
#                   "Number_people_maintenance", "Telephone", "foreign_worker", "label")
#
#german <- german %>% 
#  dummy_cols(select_columns = c("Status_existing_checking_account", "Credit_history",
#                                "Purpose", "Savings_account_bonds", "Present_employment_since",
#                                "Personal_status_and_sex", "Other_debtors_guarantors", "Property",
#                                "Other_installment_plans", "Housing", "Job", "Telephone", "foreign_worker"), remove_first_dummy = TRUE, remove_selected_columns = TRUE)
#
#german_dummies <- german_dummies %>%
#  mutate(Duration_month = Winsorize(Duration_month)) %>%
#  mutate(amount = Winsorize(amount)) %>%
#  mutate(Installment_rate_percentage_disposable_income = Winsorize(Installment_rate_percentage_disposable_income)) %>%
#  mutate(Present_residence_since = Winsorize(Present_residence_since)) %>%
#  mutate(Age = Winsorize(Age)) %>%
#  mutate(existing_credits = Winsorize(existing_credits)) %>%
#  mutate(Number_people_maintenance = Winsorize(Number_people_maintenance))
#  
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


#GMSC
kaggle <- read_csv("data/GiveMeSomeCredit/cs-training.csv")

kaggle <- kaggle %>%
  mutate_at(vars(SeriousDlqin2yrs), as.factor) %>%
  rename("label" = "SeriousDlqin2yrs") %>%
  mutate(label = factor(label, 
                        labels = make.names(levels(label))))

#add flags for missing values
kaggle$MonthlyIncome_flag <- as.factor(ifelse(((kaggle$MonthlyIncome == 0)|is.na(kaggle$MonthlyIncome)), 1, 0))
kaggle$NumberOfDependents_flag <- as.factor(ifelse(is.na(kaggle$NumberOfDependents), 1, 0))


save(kaggle, file = "C:/Users/simon/Documents/GitHub/Thesis/data/GOLD/kaggle.Rda")

#GMSC
thomas <- read_delim("data/02thomas/Loan_Data.csv", delim = ";", escape_double = FALSE, trim_ws = TRUE)

thomas <- thomas %>%
  mutate_at(vars(BAD, PHON), as.factor) %>%
  rename("label" = "BAD") %>%
  mutate(label = factor(label, 
                        labels = make.names(levels(label))))

thomas$YOB_flag <- ifelse(thomas$YOB == 99, 1, 0)
thomas$DHVAL_flag <- ifelse(thomas$DHVAL == 0, 1, 0)
thomas$DMORT_flag <- ifelse(thomas$DMORT == 0, 1, 0)

save(thomas, file = "C:/Users/simon/Documents/GitHub/Thesis/data/GOLD/thomas.Rda")
