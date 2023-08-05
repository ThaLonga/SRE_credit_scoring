library(readr)
german <- read_table("data/statlog+german+credit+data/german.csv", 
                       +     col_names = FALSE)

names(german) <- c("Status of existing checking account", "Duration in month", "Credit history", "Purpose", "Credit amount",
                   "Savings account/bonds", "Present employment since", "Installment rate in percentage of disposable income",
                   "Personal status and sex", "Other debtors / guarantors", "Present residence since", "Present residence since",
                   "Property", "Age", "Other installment plans", "Housing", "Number of existing credits at this bank", "Job",
                   "Number of people being liable to provide maintenance for", "Telephone", "foreign worker")
