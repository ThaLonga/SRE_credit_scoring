# main

#conventions:
# target variable = label
# indepentent variables = all others

#levels should be X1 and X2
#evaluation function for caret tuning from pre package
source("./src/models.R")
source("./src/hyperparameters.R")
source("./src/BigSummary.R")
source("./src/data_loader.R")
datasets <- load_data()

#(AUCROC, Brier, partialGini)
metric = "partialGini"
µnr_repeats = 5
outerfolds = 2


# create empty dataframe metric_results with columns: (dataset, repeat, fold, algorithm, metric)	
metric_results <- data.frame(
  dataset = character(),
  nr_repeat = integer(),
  nr_outer_fold = integer(),
  algorithm = character(),
  metric = double(),
  stringsAsFactors = FALSE
)


#metric_results[nrow(metric_results) + 1,] = list(dataset, 2, 1, "test", 40.485)



set.seed(123)
train_indices <- sample(1:nrow(x), 0.8 * nrow(x))
x_train <- (x[train_indices, ])
x_test <- (x[-train_indices, ])
y_train <- (y[train_indices, ])
y_test <- (y[-train_indices, ])