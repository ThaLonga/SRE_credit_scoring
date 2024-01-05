# MODELLING

######
#setup
######
#import
if (!require("pacman")) install.packages("pacman") ; require("pacman")
p_load(glmnet, glmnetUtils, mgcv, tidyverse, xgboost, DiagrammeR, stringr, tictoc, parallel, pROC, earth, Matrix, pre, caret, parsnip)
source("hyperparameters.R")

#setup cluster
cl <- makeCluster(detectCores()-1)

#evaluation function for caret tuning from pre package
BigSummary <- function (data, lev = NULL, model = NULL) {
  brscore <- try(mean((data[, lev[2]] - ifelse(data$obs == lev[2], 1, 0)) ^ 2),
                 silent = TRUE)
  rocObject <- try(pROC::roc(ifelse(data$obs == lev[2], 1, 0), data[, lev[2]],
                             direction = "<", quiet = TRUE), silent = TRUE)
  if (inherits(brscore, "try-error")) brscore <- NA
  rocAUC <- if (inherits(rocObject, "try-error")) {
    NA
  } else {
    rocObject$auc
  }
  return(c(AUCROC = rocAUC, Brier = brscore))
}

# return boolean vector TRUE for numerical columns (not boolean)
get_splineworthy_columns <- function(X) {
  return(lapply(X, n_distinct)>5)
}

#save model
save_model <- function(object, name) {
  object_name <- deparse(substitute(object))
  saveRDS(object, paste("./models/",name,"_",dataset,".RDS", sep=""))
}

######

#auc, BS, 
metric = "auc"
metric_caret = ifelse(metric=="auc", "ROC", metric)
n_folds = 5

########################
# Loading & partitioning
########################

#Choose data to load

dataset = "german"

load(paste("data/GOLD/x_",dataset,".Rda", sep=""))
load(paste("data/GOLD/y_",dataset,".Rda", sep=""))

set.seed(123)
train_indices <- sample(1:nrow(x), 0.8 * nrow(x))
x_train <- (x[train_indices, ])
x_test <- (x[-train_indices, ])
y_train <- (y[train_indices, ])
y_test <- (y[-train_indices, ])


##### MOVE TO PREPROCESSING
y_train = y_train %>% as.factor()
y_test = y_test %>% as.factor()

train = as_tibble(cbind(x_train, y_train)) %>% rename(label = y_train)
test = as_tibble(cbind(x_test, y_test)) %>% rename(label = y_test)
train$label = as.factor(train$label)
test$label = as.factor(test$label)

#####################
#####################
# Linear models
#####################
#####################

#####################
# LR
#####################

LR_model_final = glm(label ~., data = train, family = "binomial")
saveRDS(LR_model_final, file="./models/LR_model_final.rds")
#####################
# LR-R
#####################

LR_R_ctrl = trainControl(method = "cv", number = n_folds, classProbs = TRUE, summaryFunction = twoClassSummary)

train = train  %>% 
        mutate(label = factor(label, 
                        labels = make.names(levels(label))))

set.seed(123)
LRR_model <- train(label ~., data = train,  method = "glmnet", trControl = LR_R_ctrl, metric = metric_caret,
                   tuneGrid = expand.grid(alpha = hyperparameters_LR_R$alpha,lambda = hyperparameters_LR_R$lambda),
                   allowParallel=TRUE)
LRR_model_final = LRR_model$finalModel
#best tune: alpha = 0.1, lambda = 0.1
#coef(LRR_model$finalModel, LRR_model$bestTune$lambda)


###################
##### LDA
###################

# Correlation based feature selection
corr_matrix <- cor(cbind(x_train, y_train))

#correlated features:
(sum(abs(corr_matrix)>0.8)-49)
#GC: 2 -> HousingA153 and PropertyA124
cor(cbind(train$HousingA153, train$PropertyA124, y_train))

train_DA <- train %>% select(-"HousingA153")
#AC:


###################
##### QDA
###################


###################
##### GAM
###################

smooth_vars = colnames(train)[get_splineworthy_columns(train)]
formula <- as.formula(
  stringr::str_sub(paste("label ~", 
        paste(ifelse(names(train) %in% smooth_vars, "s("  , ""),
              names(train),
              ifelse(names(train) %in% smooth_vars, ", k=4)",""),
              collapse = " + ")
  ), 0, -10)
)

gamtest <- gam(formula, family = "binomial", data = train) #degrees freedom error
save_model(gamtest, "gam")
###################
##### MARS
###################

mars_model = earth(y_train ~ x_train, degree=2)

train = train  %>% 
  mutate(label = factor(label, 
                        labels = make.names(levels(label))))

cv_mars <- train(
  label~.,
  data = train,
  method = "earth",
  metric = metric_caret,
  trControl = trainControl(method = "cv", number = n_folds, classProbs = TRUE, allowParallel = TRUE),
  tuneGrid = expand.grid(nprune=50, degree = hyperparameters_MARS$degree)
)

# View results
cv_mars$bestTune
##    nprune degree
## 16     56      2

cv_mars$results %>%
  filter(nprune == cv_mars$bestTune$nprune, degree == cv_mars$bestTune$degree)

###################
#####Rule ensembles
###################

fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1,
                           classProbs = TRUE, ## get probabilities, not class labels
                           summaryFunction = BigSummary, verboseIter = TRUE)

#ADJUST
preGrid <- getModelInfo("pre")[[1]]$grid(
  maxdepth = 2L:3L,
  learnrate = c(.01, .05, .1),
  penalty.par.val = c("lambda.1se", "lambda.min"),
  sampfrac = c(0.5, 0.75, 1.0))


RE_model <- pre(label ~ .,
               family = binomial,
               data = train,
               verbose = TRUE)

RE_fit <- fit_xy(RE_model, x_train, y_train)

cvpre(RE_model,
      k = 3,
      verbose = TRUE)


#testing WORKS
rule <- RE_model$rules$description[3]
# Split the rule into individual conditions
conditions <- strsplit(rule, " & ")[[1]]

# Add 'train$' before each condition
conditions_with_train <- paste("train$", conditions, sep = "")

# Combine the conditions with ' & ' separator
final_rule <- parse(text = paste(conditions_with_train, collapse = " & "))

subset_train <- train[eval(final_rule), ]



#AUC
xpreds <- x_test
xpreds$good <- y_test
prob=predict(RE_model, newdata = data.frame(x_test), type=c("response"))
xpreds$prob<-data.frame(prob)
g <- roc(unlist(good) ~ unlist(prob), data = xpreds)
plot(g)    
AUC <- g$auc
#

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


SRE_rules_splines() <- function(nrounds, )
#####
# rules
#####
# convert the train and test data into xgboost matrix type.
xgboost_train = xgb.DMatrix(data=x_train, label=y_train)

xgb_model <- xgboost(data = xgboost_train, max.depth = 3, nrounds = 20)

tree_dump <- xgb.dump(xgb_model)




#####
#ALTERNATIVE (better)
xgb.plot.tree(model = xgb_model, trees = 0)

model_trees <- xgb.model.dt.tree(model = xgb_model)
model_trees <- as_tibble(model_trees[model_trees$Feature!="Leaf"]) %>% select(Tree, Node, Feature, Split)
model_trees %>% filter(Tree==0)
model_trees_array <- split(model_trees, model_trees$Tree)


# Function to generate rules
generate_rules <- function(tree_data, node) {
  current_node <- subset(tree_data, Node == node)
  rule <- paste(current_node$Node, collapse = "-")
  if (nrow(current_node) > 0) {
    left_child <- current_node$Node[1] * 2 + 1
    right_child <- current_node$Node[1] * 2 + 2
    rule_left <- generate_rules(tree_data, left_child)
    rule_right <- generate_rules(tree_data, right_child)
    return(c(rule, rule_left, rule_right))
  } else {
    return(rule)
  }
}

# Generate rules starting from the root node (0)
rules <- generate_rules((model_trees %>% filter(Tree==0)), 0)

# Print the rules
print(rules)
#####


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
basetable <- data.frame(1:800)
basetable_test <- data.frame(1:800)

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

basetable = basetable[,-1]
basetable_test = basetable_test[,-1]

#scale
supports_rules <- colMeans(basetable)
stdevs_rules <- sqrt(supports_rules*(1-supports_rules))
basetable <- sweep(basetable, 2, stdevs_rules, FUN = "/")
basetable_test <- sweep(basetable_test, 2, stdevs_rules, FUN = "/")

x_train_stdevs <- sapply(x_train, sd)
x_train_scaled = 0.4*sweep(x_train, 2, x_train_stdevs, FUN = "/")
x_test_scaled = 0.4*sweep(x_test, 2, x_train_stdevs, FUN = "/")

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

#scaling
basetable



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
