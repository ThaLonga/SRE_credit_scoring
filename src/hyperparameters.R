# Hyperparameters

#Lasso
hyperparameters_LR_R <- list(
  alpha = c(0,.2,.4,.6,.8,1),
  lambda = c(0, 0.001,0.01,0.1,1,10)
)

hyperparameters_LR_R_tidy <- expand.grid(list(
  mixture = c(0,.2,.4,.6,.8,1),
  penalty = c(0, 0.001,0.01,0.1,1,10)
))


#GAM
#splines for numerical

#LDA


#QDA


#CTREE
hyperparameters_CTREE <- list(
  mincriterion = 1-c(0.05,0.1,0.2,0.3,0.4)
)

#RF
# Lessmann 2015
hyperparameters_RF <- list(
  mtry = sqrt(ncol(train_bake_x)*c(0.1,0.25,0.5,1,2,4)),
  splitrule = "gini",
  min.node.size = 1
)

#XGB
# B2Boost  and Lessmann
hyperparameters_XGB <- expand.grid(list(
  nrounds = c(10,50,100,250,500,1000), #L
  eta = c(0.001, 0.01, 0.1, 0.2, 0.5), #B
  gamma = c(0.5, 1, 1.5, 2), #B
  max_depth = 6, #default
  colsample_bytree = 1, #default
  min_child_weight = 1, #default
  subsample = 1 #default
))

hyperparameters_XGB_tidy <- expand.grid(list(
  trees = c(10,50,100,250,500,1000), #L
  learn_rate = c(0.001, 0.01, 0.1, 0.2, 0.5), #B
  loss_reduction = c(0.5, 1, 1.5, 2), #B
  tree_depth = 6 #default
  #colsample_bytree = 1, #default
  #min_child_weight = 1, #default
  #subsample = 1 #default
))



#LightGBM eventueel

#hyperparameters_
#hyperparameters_
#hyperparameters_
#hyperparameters_