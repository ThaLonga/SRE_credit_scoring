# Hyperparameters

#Lasso
hyperparameters_LR_R_tidy <- expand.grid(list(
  mixture = c(0,.2,.4,.6,.8,1),
  penalty = seq(0.001,1, length.out = 100)
))

hyperparameters_SRE_tidy <- expand.grid(list(
  penalty = seq(0.001,1, length.out = 1000)
))

#CTREE
hyperparameters_CTREE <- list(
  mincriterion = 1-c(0.05,0.1,0.2,0.3,0.4)
)

#RF
# Lessmann 2015
hyperparameters_RF_tidy <- expand.grid(list(
  trees = c(100,250,500,750,1000),
  mtry = sqrt(ncol(train_bake_x)*c(0.1,0.25,0.5,1,2,4)),
  min_n = 1
))

#XGB
#DL for CS
hyperparameters_XGB_tidy <- crossing(
  trees = c(50,100, 150), #L
  learn_rate = c(0.3, 0.4), #B
  tree_depth = c(1,2,3), #default
  mtry = round(ncol(train_bake_x)*c(0.6,0.8)), #default
  sample_size = c(0.5, 0.75, 1)
)

hyperparameters_LGBM_tidy <- crossing(
  trees = c(50,100, 150), #L
  learn_rate = c(0.3, 0.4), #B
  tree_depth = c(1,2,3),
  mtry = round(ncol(train_bake_x)*c(0.6,0.8))
)

preGrid <- getModelInfo("pre")[[1]]$grid( 
  maxdepth = c(1,2,3),
  learnrate = c(.3, .5),
  penalty.par.val = c("lambda.1se"), # λand γ combination yielding the sparsest solution within 1 standard error of the error criterion of the minimum is returned
  sampfrac = 1,
  use.grad = TRUE,
  mtry = round(sqrt(ncol(train_bake_x)*c(0.1,0.25,0.5,1,2,4)))
)
