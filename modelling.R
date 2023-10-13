# MODELLING
if (!require("pacman")) install.packages("pacman") ; require("pacman")
p_load(glmnet, tidyverse, xgboost, DiagrammeR, stringr, tictoc, parallel, pROC)

#setup cluster
cl <- makeCluster(detectCores()-1)

########################
# Loading & partitioning
########################

#Choose data to load
load("data/x_german.Rda")
load("data/y_german.Rda")

set.seed(123)
train_indices <- sample(1:nrow(x), 0.8 * nrow(x))
x_train <- x[train_indices, ]
x_test <- x[-train_indices, ]
y_train <- y[train_indices, ]
y_test <- y[-train_indices, ]

#####################
# Logistic regression
#####################


## Ridge Regression to create the Adaptive Weights Vector
set.seed(123)
cv.ridge <- cv.glmnet(x_train, y_train, family='binomial', alpha=0, parallel=TRUE, standardize=TRUE)

# weights = 1/absolute value of ridge coefficients
w3 <- 1/abs(matrix(coef(cv.ridge, s=cv.ridge$lambda.min)
                   [, 1][2:(ncol(x_train)+1)] ))^1 ## Using gamma = 1
w3[w3[,1] == Inf] <- 999999999 ## Replacing values estimated as Infinite for 999999999

# adaptive Lasso
set.seed(123)
cv.lasso <- cv.glmnet(x_train, y_train, family='binomial', alpha=1, parallel=TRUE, standardize=TRUE, type.measure='auc', penalty.factor=w3)

####
# evaluation
####
###################
#####Rule ensembles
###################
#adaptive lasso

######
#SRE
######

#####
# linear terms
#####

#SRE <- function()

# Winsorization
lintable <- x_train
lintable_test <- x_test
for(c in 1:ncol(x_train)) {
  d_min <- quantile(lintable[,c], probs = 0.025)
  d_plus <- quantile(lintable[,c], probs = 0.975)
  for(r in 1:nrow(x_train)) {
    lintable[r,c] <- min(d_plus, max(d_min, lintable[r,c]))
  }
  for(r in 1:nrow(x_test)) {
    lintable_test[r,c] <- min(d_plus, max(d_min, lintable_test[r,c]))
  }
}



#####
# rules
#####
# convert the train and test data into xgboost matrix type.
xgboost_train = xgb.DMatrix(data=x_train, label=y_train)

xgb_model <- xgboost(data = xgboost_train, max.depth = 2, nrounds = 20)

tree_dump <- xgb.dump(xgb_model)


# Parse the tree dump to extract rules
extract_rules <- function(tree_dump) {
  rules <- list()
  for (i in 1:length(tree_dump)) {
    tree <- tree_dump[[i]]
    # Split the tree dump into lines and process each line
    lines <- strsplit(tree, "\n")[[1]]
    rule <- list()
    for (line in lines) {
      # Extract rule information from each line (modify the parsing logic as needed)
      # Example parsing logic: extracting condition, feature, threshold, and leaf value
      # Rule format: "feature_name < threshold"
      # Leaf node format: "leaf=value"
      if (grepl("<", line)) {
        condition <- gsub(".*\\[", "", line)
        condition <- gsub("\\].*", "", condition)
        rule[length(rule) + 1] <- condition
      } #else if (grepl("leaf", line)) {
        #leaf <- gsub(".*leaf=", "", line)
        #rule[length(rule) + 1]<- paste("leaf=", leaf)
      #}
      else if (grepl("booster", line)) {
        treenr <- gsub(".*\\[", "", line)
        treenr <- gsub("\\].*", "", treenr)
        rule[length(rule) + 1]<- paste("tree ", treenr)
      }
    }
    rules[[i]] <- rule
  }
  
  extracted_rules_cleaned <- list()
  for(i in 1:length(rules)) {
    if(!is.null(unlist(rules[[i]][1]))) {
      value <- unlist(rules[[i]][1])
      extracted_rules_cleaned <- append(extracted_rules_cleaned, value)
    }
    
  }
  return(extracted_rules_cleaned)
}

# Extract rules from the tree dump
extracted_rules <- extract_rules(tree_dump) #now create features

rule_matrix <- matrix(nrow = length(extracted_rules), ncol = 3)
for(l in 1:length(extracted_rules)) {
  if(grepl("tree", extracted_rules[l])) {
    tree_nr <- gsub("tree  ", "", extracted_rules[l])
  }
  else if(grepl("<", extracted_rules[l])) {
    rule_matrix[l, 2] <- str_extract(extracted_rules[l], "(?<=f)(.*?)(?=<)")
    rule_matrix[l, 3] <- str_extract(extracted_rules[l], "(?<=<)(.*)")
  }
  rule_matrix[l,1] <- tree_nr
}


#add rules to basetable
basetable <- data.frame(x_train)
basetable_test <- data.frame(x_test)

#names are given as such: "factor_10_<_0.5_and_factor_1_<_22.5
c<-1
for(i in which(is.na(rule_matrix[,2]))) {
  if(!is.na(rule_matrix[i+1,2])) {
    basetable[,paste("T", c, "I(factor", (as.numeric(rule_matrix[i+1,2])+1), "<", as.numeric(rule_matrix[i+1,3]), ")", sep = "_")] <- x_train[,as.numeric(rule_matrix[i+1,2])+1]<as.numeric(rule_matrix[i+1,3], ")") 
    basetable_test[,paste("T", c, "I(factor", (as.numeric(rule_matrix[i+1,2])+1), "<", as.numeric(rule_matrix[i+1,3]), ")", sep = "_")] <- x_test[,as.numeric(rule_matrix[i+1,2])+1]<as.numeric(rule_matrix[i+1,3], ")") 
    }
  if(!is.na(rule_matrix[i+2,2])) {
    basetable[,paste("T", c, "I(factor", (as.numeric(rule_matrix[i+1,2])+1), "<", as.numeric(rule_matrix[i+1,3]), "and_factor", (as.numeric(rule_matrix[i+2,2])+1), "<", as.numeric(rule_matrix[i+2,3]), ")", sep = "_")] <- (x_train[,as.numeric(rule_matrix[i+1,2])+1]<as.numeric(rule_matrix[i+1,3]) & x_train[,as.numeric(rule_matrix[i+2,2])+1]<as.numeric(rule_matrix[i+2,3]))
    basetable_test[,paste("T", c, "I(factor", (as.numeric(rule_matrix[i+1,2])+1), "<", as.numeric(rule_matrix[i+1,3]), "and_factor", (as.numeric(rule_matrix[i+2,2])+1), "<", as.numeric(rule_matrix[i+2,3]), ")", sep = "_")] <- (x_test[,as.numeric(rule_matrix[i+1,2])+1]<as.numeric(rule_matrix[i+1,3]) & x_test[,as.numeric(rule_matrix[i+2,2])+1]<as.numeric(rule_matrix[i+2,3]))
    }
  if(!is.na(rule_matrix[i+3,2])) {
    basetable[,paste("T", c, "I(factor", (as.numeric(rule_matrix[i+1,2])+1), "<", as.numeric(rule_matrix[i+1,3]), "and_factor", (as.numeric(rule_matrix[i+3,2])+1), "<", as.numeric(rule_matrix[i+3,3]), ")", sep = "_")] <- (x_train[,as.numeric(rule_matrix[i+1,2])+1]<as.numeric(rule_matrix[i+1,3]) & x_train[,as.numeric(rule_matrix[i+3,2])+1]<as.numeric(rule_matrix[i+3,3]))
    basetable_test[,paste("T", c, "I(factor", (as.numeric(rule_matrix[i+1,2])+1), "<", as.numeric(rule_matrix[i+1,3]), "and_factor", (as.numeric(rule_matrix[i+3,2])+1), "<", as.numeric(rule_matrix[i+3,3]), ")", sep = "_")] <- (x_test[,as.numeric(rule_matrix[i+1,2])+1]<as.numeric(rule_matrix[i+1,3]) & x_test[,as.numeric(rule_matrix[i+3,2])+1]<as.numeric(rule_matrix[i+3,3]))
    }
  c<-c+1
}
#for every 4 rows: 
#row 1: tree index
#row 2: rule 1
#row 2 and row 3: rule 2
#row 2 and row 4: rule 3


# Plot specific decision tree
#xgb.plot.tree(model = xgb_model, trees = 0)


get_numeric_columns <- function(dataframe) {
  return(which(sapply(data.frame(dataframe), function(col) any(col != 0 & col != 1))))
}


#LOOCV when dataset not large

for(i in get_numeric_columns(x_train)) {
  #Smoothing splines only for variables with at least 4 distinct values 
  if(length(unique(x_train[,i])) > 3) {
  smooth_spline <- smooth.spline(x_train[,i], y_train, cv = FALSE)
  basetable[,paste("Spline_", i)] <- predict(smooth_spline, x_train[,i])$y
  basetable_test[,paste("Spline_", i)] <- predict(smooth_spline, x_test[,i])$y
  }
}


basetable_num <- data.frame(sapply(basetable, as.numeric))
basetable_num_test <- data.frame(sapply(basetable_test, as.numeric))

## Ridge Regression to create the Adaptive Weights Vector
set.seed(123)
cv.ridge <- cv.glmnet(as.matrix(basetable_num), y_train, family='binomial', alpha=0, parallel=TRUE, standardize=TRUE)

# weights = 1/absolute value of ridge coefficients
w3 <- 1/abs(matrix(coef(cv.ridge, s=cv.ridge$lambda.min)
                   [, 1][2:(ncol(basetable_num)+1)] ))^1 ## Using gamma = 1
w3[w3[,1] == Inf] <- 999999999 ## Replacing values estimated as Infinite for 999999999

# adaptive Lasso
set.seed(123)
cv.lasso <- cv.glmnet(as.matrix(basetable_num), y_train, family='binomial', alpha=1, parallel=TRUE, standardize=TRUE, type.measure='auc', penalty.factor=w3)

#predict(cv.lasso, type="coef")

#####
# evaluation
#####

#based on Lessmann et al. 3 performance measures

#AUC
xpreds <- basetable_num_test
xpreds$good <- y_test
prob=predict(cv.lasso, newx = as.matrix(basetable_num_test), type=c("response"))
xpreds$prob<-data.frame(prob)
g <- roc(unlist(good) ~ unlist(prob), data = xpreds)
plot(g)    
AUC <- g$auc
#


#PG

#𝐺𝑖𝑛𝑖=2*partial 𝐴𝑈𝐶/(a+b)(b-a) − 1

(2*auc(good ~ prob, data = as.data.frame(xpreds), partial.auc = c(0,0.5))/((0+0.5)*(0.5-0)))-1
2*(AUC-0.5)
#BS
#accuracy: closer to 0 = better (1/N)*sum((f-o)²)

(BS <- sum((prob-y_test)^2)/nrow(y_test))
#0.1617

# plots
plot(cv.lasso)
plot(cv.lasso$glmnet.fit, xvar="lambda", label=TRUE)
abline(v = log(cv.lasso$lambda.min))
abline(v = log(cv.lasso$lambda.1se))
coef(cv.lasso, s=cv.lasso$lambda.1se)
coef <- coef(cv.lasso, s='lambda.1se')
selected_attributes <- (coef@i[-1]+1) ## Considering the structure of the data frame dataF as shown earlier



#adaptive lasso

stopCluster(cl)