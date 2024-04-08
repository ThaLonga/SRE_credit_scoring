# Hyperparameters

#Lasso
hyperparameters_LR_R <- list(
  alpha = c(0,.2,.4,.6,.8,1),
  lambda = seq(0.001,1, length.out = 100)
)

hyperparameters_LR_R_tidy <- expand.grid(list(
  mixture = c(0,.2,.4,.6,.8,1),
  penalty = seq(0.001,1, length.out = 100)
))

hyperparameters_SRE_tidy <- expand.grid(list(
  penalty = seq(0.001,1, length.out = 1000)
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
hyperparameters_XGB <- list(
  nrounds = c(10,50,100,250,500), #L
  eta = c(0.01, 0.1, 0.2, 0.5), #B
  gamma = c(0.5, 1, 1.5, 2), #B
  max_depth = 6, #default
  colsample_bytree = 1, #default
  min_child_weight = 1, #default
  subsample = 1 #default
)

hyperparameters_XGB_tidy <- crossing(
  trees = c(10,50,100,250,500,1000), #L
  learn_rate = c(0.01, 0.1, 0.2, 0.5), #B
  loss_reduction = c(0.5, 1, 1.5, 2), #B
  tree_depth = 6 #default
  #colsample_bytree = 1, #default
  #min_child_weight = 1, #default
  #subsample = 1 #default
)



#LightGBM eventueel

preGrid <- getModelInfo("pre")[[1]]$grid( 
  maxdepth = c(2,3),
  learnrate = c(.01, .05, .1),
  penalty.par.val = c("lambda.1se"), # λand γ combination yielding the sparsest solution within 1 standard error of the error criterion of the minimum is returned
  sampfrac = 1,
  use.grad = TRUE
  #mtry = sqrt(ncol(train_bake_x)*c(0.1,0.25,0.5,1,2,4))
) #adaptive lasso with ridge weights
# !! nlambda by default 100 models 

#hyperparameters_
#hyperparameters_
#hyperparameters_
#hyperparameters_