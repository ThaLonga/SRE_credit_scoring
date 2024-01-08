# main

#conventions:
# target variable = label
# indepentent variables = all others

#levels should be X1 and X2
#evaluation function for caret tuning from pre package
source("./src/hyperparameters.R")
source("./src/BigSummary.R")

#(AUCROC, Brier, partialGini)
metric = "partialGini"