if (!require("pacman")) install.packages("pacman") ; require("pacman")
p_load(tidyverse, fastDummies, dplyr, DescTools, readxl)

source('./src/preprocessing_functions.R')





#German
german <- read_table("data/statlog+german+credit+data/german.data", col_names = FALSE)

## change 1,2 to 0,1
german$X21 <- as.factor(ifelse(german$X21==1, 0, 1))
german <- german  %>% rename("label" = X21) %>%
  mutate(label = factor(label, 
                        labels = make.names(levels(label))))

save(german, file = "data/GOLD/german.Rda")


#Australian
australian <- read_table("data/statlog+australian+credit+approval/australian.dat", col_names = FALSE)

australian <- australian %>%
  mutate_at(vars(X1, X4, X5, X6, X8, X9, X11, X12, X15), as.factor) %>%
  rename("label" = "X15") %>%
  mutate(label = factor(label, 
                        labels = make.names(levels(label))))

save(australian, file = "data/GOLD/australian.Rda")

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

save(HMEQ, file = "data/GOLD/hmeq.Rda")



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
  dplyr::select(-...1)

#add flags for missing values
kaggle$MonthlyIncome_flag <- as.factor(ifelse(((kaggle$MonthlyIncome == 0)|is.na(kaggle$MonthlyIncome)), 1, 0))
kaggle$NumberOfDependents_flag <- as.factor(ifelse(is.na(kaggle$NumberOfDependents), 1, 0))

save(kaggle, file = "data/GOLD/kaggle.Rda")


#TH02
thomas <- read_delim("data/02thomas/Loan_Data.csv.txt", delim = ";", escape_double = FALSE, trim_ws = TRUE)

thomas <- thomas %>%
  mutate_at(vars(BAD, PHON), as.factor) %>%
  rename("label" = "BAD") %>%
  mutate(label = factor(label, 
                        labels = make.names(levels(label))))

thomas$YOB_flag <- as.factor(ifelse(thomas$YOB == 99, 1, 0))
thomas$DHVAL_flag <- as.factor(ifelse(thomas$DHVAL == 0, 1, 0))
thomas$DMORT_flag <- as.factor(ifelse(thomas$DMORT == 0, 1, 0))

save(thomas, file = "data/GOLD/thomas.Rda")

#LC
LC <- read_delim("data/LC/LC.csv", delim = ",", escape_double = FALSE, trim_ws = TRUE, col_names = FALSE)

LC <- LC %>%
  mutate_at(vars(X2, X11), as.factor) %>%
  rename("label" = "X11") %>%
  mutate(label = factor(label, 
                        labels = make.names(levels(label))))

save(LC, file = "data/GOLD/LC.Rda")

#TC
TC <- read_excel("data/default+of+credit+card+clients/default of credit card clients.xls", trim_ws = TRUE, range=cell_rows(2:30002), col_names = TRUE)
#33% sample
TC <- TC[sample(nrow(TC), round(1/3*nrow(TC))), ]
TC <- TC %>%
  mutate_all(as.integer) %>%
  rename("label" = "default payment next month") %>%
  mutate_at(vars(label), as.factor) %>%
  mutate(label = factor(label, 
                        labels = make.names(levels(label))))

save(TC, file = "data/GOLD/TC.Rda")

#PAKDD
PAKDD <- read_delim("data/PAKDD 2010/PAKDD.csv", delim = ";", escape_double = FALSE, trim_ws = TRUE)
PAKDD <- PAKDD %>% dplyr::select(-Column2, -Column11, -Column12, -Column14, -Column15, -Column16, -Column18, -Column35, -Column36, -Column37, -Column39, -Column52, -Column53)
PAKDD[PAKDD=="NULL"]<-NA
PAKDD$Column4[PAKDD$Column4=="0"]<-NA
PAKDD$Column7[PAKDD$Column7=="N"]<-NA

PAKDD <- PAKDD %>%
  mutate_at(vars(Column3, Column6, Column7, Column8, Column13, Column17, Column19, Column20, Column21, Column25, Column26, Column27, Column28, Column29, Column34, Column38, Column41, Column42, Column43, Column44, Column45, Column46, Column47, Column48, Column49, Column50, Column54), as.factor) %>%
  rename("label" = "Column54") %>%
  mutate(label = factor(label, 
                        labels = make.names(levels(label))))

#add flags for missing values
columns_to_flag <- colnames(PAKDD)[unlist(lapply(PAKDD, function(x) sum(is.na(x))))>0]
for (col in columns_to_flag) {
  flag_col_name <- paste0(col, "_FLAG")
  PAKDD[[flag_col_name]] <- as.factor(ifelse(is.na(PAKDD[[col]]), 1, 0))
}

save(PAKDD, file = "data/GOLD/PAKDD.Rda")

#Indessa
BF <- read_delim("data/indessa.csv", delim = ",", escape_double = FALSE, trim_ws = TRUE)

set.seed(111)
BF <- BF[sample(nrow(BF), round(1/20*nrow(BF))), ]

BF <- BF %>% dplyr::select(-member_id, -funded_amnt, -funded_amnt_inv, -grade, -emp_title, -title, -zip_code, -addr_state)
BF$emp_length[BF$emp_length=="n/a"]<-NA
BF$emp_length[BF$emp_length=="< 1 year"]<-0
BF$emp_length<- readr::parse_number(BF$emp_length)
BF$mths_since_last_delinq[is.na(BF$mths_since_last_delinq)]<-0
BF$mths_since_last_record[is.na(BF$mths_since_last_record)]<-0
BF$mths_since_last_major_derog[is.na(BF$mths_since_last_major_derog)]<-0
BF$last_week_pay <- as.numeric(str_extract(BF$last_week_pay,"\\d+(?=th)"))

BF <- BF %>%
  mutate_at(vars(term, sub_grade, home_ownership, verification_status, pymnt_plan, purpose, initial_list_status, application_type, verification_status_joint), as.factor) %>%
  rename("label" = "loan_status") %>%
  mutate(label = factor(label, 
                        labels = make.names(levels(as.factor(label)))))

#add flags for missing values
columns_to_flag <- c("batch_enrolled", "desc")
for (col in columns_to_flag) {
  flag_col_name <- paste0(col, "_FLAG")
  BF[[flag_col_name]] <- as.factor(ifelse(is.na(BF[[col]]), 1, 0))
}

BF <- BF %>% dplyr::select(-batch_enrolled, -desc)

save(BF, file = "data/GOLD/BF.Rda")


#FM
FM <- read_delim("data/2024Q2/2024Q2.csv", delim = ";", escape_double = FALSE, trim_ws = TRUE)
set.seed(111)
FM <- FM[sample(nrow(FM), round(1/20*nrow(FM))), ]

#remove IDs, X5, X6 because redundant
# X8 because corrent & original interest rate are equal for all observations
# X13, X14, X18 because dates
# X15 because loan age not a priori known
# X20, very high correlation with X19
FM <- FM %>% dplyr::select(-X1, -X2, -X5, -X6, -X8, -X13, -X14, -X15, -X18, -X20)
FM <- FM %>% select_if(~mean(is.na(.)) < 0.2)
FM <- FM %>% select_if(~length(unique(.)) > 1)

#create dummy for loan seller
FM$X4 <- ifelse(
  FM$X4=="Other",
  1,
  0   
)

FM$label <- as.factor(ifelse(
  is.na(FM$X40),                # Check if the value is NA
  FM$X39,                       # Impute with value from X39 if NA
  ifelse(
    grepl("[^X0]", FM$X40)|FM$X101=="N",     # Check if the string contains characters other than 'X' and '0'
    1,                          # Set to 1 if other characters are present
    0                           # Otherwise, set to 0
  )
))

#remove delinquencies, remove property state, metropolitan statistical area and zip code
FM <- FM %>% dplyr::select(-X39, -X40, -X30, -X31, -X32, -X41, -X73, -X101)

FM <- FM %>%
  mutate_at(vars(X3, X25, X26, X27, X29, X78, X80, X85, X86), as.factor) %>%
  mutate(label = factor(label, 
                        labels = make.names(levels(label))))

#add flags for missing values
columns_to_flag <- colnames(FM)[unlist(lapply(FM, function(x) sum(is.na(x))))>0]
for (col in columns_to_flag) {
  flag_col_name <- paste0(col, "_FLAG")
  FM[[flag_col_name]] <- as.factor(ifelse(is.na(FM[[col]]), 1, 0))
}

save(FM, file = "data/GOLD/FM.Rda")