# Hyperparameters

#Lasso
hyperparameters_LR_R_tidy <- expand.grid(list(
  mixture = c(0,.2,.4,.6,.8,1),
  penalty = seq(0.001,1, length.out = 200)
))

hyperparameters_SRE_tidy <- expand.grid(list(
  penalty = seq(0.001,1, length.out = 200)
))

#CTREE
hyperparameters_CTREE <- list(
  mincriterion = 1-c(0.05,0.1,0.2,0.3,0.4)
)

#RF
# Lessmann 2015
hyperparameters_RF_tidy <- expand.grid(list(
  trees = c(100),
  mtry = ceiling(sqrt(ncol(train_bake_x)*c(0.5,1,2,4))),
  min_n = 1
))

#XGB
#DL for CS
hyperparameters_XGB_tidy <- crossing(
  trees = c(100), #L
  learn_rate = c(0.3, 0.4, 0.5), #B
  tree_depth = c(10), #default
  sample_size = c(0.5, 0.75, 1),
  loss_reduction = c(0.01, 0.1, 1, 10)
)
#meer lr en loss reduct

hyperparameters_LGBM_tidy <- crossing(
  trees = c(100), #L
  learn_rate = c(0.3, 0.4, 0.5), #B
  tree_depth = c(10),
  sample_size = c(0.5, 0.75, 1),
  loss_reduction = c(0.01, 0.1, 1, 10)
)

preGrid_boosting <- getModelInfo("pre")[[1]]$grid( 
  maxdepth = c(10),
  learnrate = c(.3, .4, .5),
  penalty.par.val = c("lambda.min"), # λand γ combination yielding the sparsest solution within 1 standard error of the error criterion of the minimum is returned
  sampfrac = 1,
  use.grad = TRUE,
  mtry = ceiling(sqrt(ncol(train_bake_x)*c(0.5,1,2,4))),
)

preGrid_RF <- getModelInfo("pre")[[1]]$grid( 
  maxdepth = c(10),
  penalty.par.val = c("lambda.min"), # λand γ combination yielding the sparsest solution within 1 standard error of the error criterion of the minimum is returned
  sampfrac = 1,
  #learnrate = 0,
  use.grad = FALSE,
  mtry = ceiling(sqrt(ncol(train_bake_x)*c(0.5,1,2,4))),
)

preGrid_bag <- getModelInfo("pre")[[1]]$grid( 
  maxdepth = c(10),
  learnrate = 0,
  sampfrac = 1,
  penalty.par.val = c("lambda.min"), # λand γ combination yielding the sparsest solution within 1 standard error of the error criterion of the minimum is returned
  #use.grad = FALSE,
)


#####
#parameters from De Bock

#RF
# Lessmann 2015
hyperparameters_RF_DB <- expand.grid(list(
  trees = 100,
  mtry = sqrt(ncol(train_bake_x)*c(0.1,0.25,0.5,1,2,4)),
  min_n = 1
))

preGrid_DB <- getModelInfo("pre")[[1]]$grid( 
  maxdepth = 10,
  learnrate = c(.3, .5),
  penalty.par.val = c("lambda.1se"), # λand γ combination yielding the sparsest solution within 1 standard error of the error criterion of the minimum is returned
  sampfrac = 1,
  use.grad = TRUE,
  mtry = round(sqrt(ncol(train_bake_x)*c(0.25,0.5,1,2,4)))
)
