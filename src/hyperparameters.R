# Hyperparameters

#Lasso
hyperparameters_LR_R = list(
  alpha = c(.1,.2,.3,.4,.5,.6,.7,.8,.9,1),
  lambda = c(0.001,0.01,0.1,1,10)
)


#GAM

#MARS
hyperparameters_MARS = list(
  degree = c(1:3),
  nprune = c(seq(2, 50, length.out = 5))
)

#hyperparameters_
#hyperparameters_
#hyperparameters_
#hyperparameters_