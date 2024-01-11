# Hyperparameters

#Lasso
hyperparameters_LR_R <- list(
  alpha = c(0,.2,.4,.6,.8,1),
  lambda = c(0, 0.001,0.01,0.1,1,10)
)


#GAM
#splines for numerical

#LDA


#QDA


#CTREE
hyperparameters_CTREE <- list(
  mincriterion = 1-c(0.05,0.1,0.2,0.3,0.4)
)

#RF
hyperparameters_RF <- list(
  ntrees = c(100,250,500,750,1000),
  mtry = sqrt(ncol(train_bake_x)*c(0.1,0.25,0.5,1,2,4))
)

#hyperparameters_
#hyperparameters_
#hyperparameters_
#hyperparameters_