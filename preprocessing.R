if (!require("pacman")) install.packages("pacman") ; require("pacman")
p_load(tidyverse, fastDummies, dplyr, DescTools)

source('./src/preprocessing_functions.R')

#German
german <- read_table("data/statlog+german+credit+data/german.csv", col_names = FALSE)

names(german) <- c("Status_existing_checking_account", "Duration_month", "Credit_history", "Purpose", "amount",
                   "Savings_account_bonds", "Present_employment_since", "Installment_rate_percentage_disposable_income",
                   "Personal_status_and_sex", "Other_debtors_guarantors", "Present_residence_since",
                   "Property", "Age", "Other_installment_plans", "Housing", "existing_credits", "Job",
                   "Number_people_maintenance", "Telephone", "foreign_worker", "target")

german_dummies <- german %>% 
  dummy_cols(select_columns = c("Status_existing_checking_account", "Credit_history",
                                "Purpose", "Savings_account_bonds", "Present_employment_since",
                                "Personal_status_and_sex", "Other_debtors_guarantors", "Property",
                                "Other_installment_plans", "Housing", "Job", "Telephone", "foreign_worker"), remove_first_dummy = TRUE, remove_selected_columns = TRUE)

german_dummies <- german_dummies %>%
  mutate(Duration_month = Winsorize(Duration_month)) %>%
  mutate(amount = Winsorize(amount)) %>%
  mutate(Installment_rate_percentage_disposable_income = Winsorize(Installment_rate_percentage_disposable_income)) %>%
  mutate(Present_residence_since = Winsorize(Present_residence_since)) %>%
  mutate(Age = Winsorize(Age)) %>%
  mutate(existing_credits = Winsorize(existing_credits)) %>%
  mutate(Number_people_maintenance = Winsorize(Number_people_maintenance))
  
# change 1,2 to 0,1
german_with_dummies$target <- german_dummies$target-1
x_german <- german_dummies %>% 
  select(-target) %>%
  as.matrix()
y_german <- german_dummies %>% 
  select(target) %>%
  as.matrix()

save(x_german, file = "C:/Users/simon/Documents/GitHub/Thesis/data/GOLD/x_german.Rda")
save(y_german, file = "C:/Users/simon/Documents/GitHub/Thesis/data/GOLD/y_german.Rda")


#Australian
australian <- read_table("data/statlog+australian+credit+approval/australian.dat", col_names = FALSE)

australian_dummies <- australian %>% 
  dummy_cols(select_columns = c("X1" , "X4" , "X5" , "X6" , "X8" , "X9" , "X11" , "X12"), remove_first_dummy = TRUE, remove_selected_columns = TRUE) %>%
  rename("target" = "X15")

australian_dummies <- australian_dummies %>%
  mutate(X2 = Winsorize(X2)) %>%
  mutate(X3 = Winsorize(X3)) %>%
  mutate(X7 = Winsorize(X7)) %>%
  mutate(X10 = Winsorize(X10)) %>%
  mutate(X13 = Winsorize(X13)) %>%
  mutate(X14 = Winsorize(X14))
  
x_australian <- australian_dummies %>% 
  select(-target) %>%
  as.matrix()
y_australian <- australian_dummies %>% 
  select(target) %>%
  as.matrix()

save(x_australian, file = "C:/Users/simon/Documents/GitHub/Thesis/data/GOLD/x_australian.Rda")
save(y_australian, file = "C:/Users/simon/Documents/GitHub/Thesis/data/GOLD/y_australian.Rda")


#GMSC
kaggle <- read_csv("data/GiveMeSomeCredit/cs-training.csv")

#no dummies #check Lessman 2015 for preprocessing
kaggle_imputed <- impute_missing_by_mean_with_dummy(kaggle)

kaggle_imputed <- kaggle_imputed %>%
  select(-...1) %>%
  rename("target" = "SeriousDlqin2yrs")

australian_dummies <- australian_dummies %>%
  mutate(X2 = Winsorize(X2)) %>%
  mutate(X3 = Winsorize(X3)) %>%
  mutate(X7 = Winsorize(X7)) %>%

x_kaggle <- kaggle_imputed %>% 
  select(-target) %>%
  as.matrix()
y_kaggle <- kaggle_imputed %>% 
  select(target) %>%
  as.matrix()

save(x_kaggle, file = "C:/Users/simon/Documents/GitHub/Thesis/data/GOLD/x_kaggle.Rda")
save(y_kaggle, file = "C:/Users/simon/Documents/GitHub/Thesis/data/GOLD/y_kaggle.Rda")
