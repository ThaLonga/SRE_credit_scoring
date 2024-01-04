if (!require("pacman")) install.packages("pacman") ; require("pacman")
p_load(tidyverse)

german <- read_table("data/statlog+german+credit+data/german.csv", col_names = FALSE)

names(german) <- c("Status_existing_checking_account", "Duration_month", "Credit_history", "Purpose", "amount",
                   "Savings_account_bonds", "Present_employment_since", "Installment_rate_percentage_disposable_income",
                   "Personal_status_and_sex", "Other_debtors_guarantors", "Present_residence_since",
                   "Property", "Age", "Other_installment_plans", "Housing", "existing_credits", "Job",
                   "Number_people_maintenance", "Telephone", "foreign_worker", "good")

german_cats_dummies <- model.matrix(~ Status_existing_checking_account + Credit_history + 
                                      Purpose + Savings_account_bonds + Present_employment_since + 
                                      Personal_status_and_sex + Other_debtors_guarantors + Property + 
                                      Other_installment_plans + Housing + Job + Telephone + 
                                      foreign_worker, german) 

german_with_dummies <- cbind(as.data.frame(german), as.data.frame(german_cats_dummies)) %>%
  select(-`(Intercept)`, -Status_existing_checking_account, -Credit_history, 
         -Purpose, -Savings_account_bonds, -Present_employment_since, 
         -Personal_status_and_sex, -Other_debtors_guarantors, -Property,
         -Other_installment_plans, -Housing, -Job, -Telephone, -foreign_worker)

# change 1,2 to 0,1
german_with_dummies$good <- german_with_dummies$good-1
x <- german_with_dummies %>% 
  select(-good) %>%
  as.matrix()
y <- german_with_dummies %>% 
  select(good) %>%
  as.matrix()

gold <- german_with_dummies

save(x, file = "C:/Users/simon/Documents/GitHub/Thesis/data/GOLD/x_german.Rda")
save(y, file = "C:/Users/simon/Documents/GitHub/Thesis/data/GOLD/y_german.Rda")
