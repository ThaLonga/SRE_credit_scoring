#########################
#####Logistic regression
#########################

if (!require("pacman")) install.packages("pacman") ; require("pacman")
p_load(glmnet, tidyverse)

## Ridge Regression to create the Adaptive Weights Vector
set.seed(123)
cv.ridge <- cv.glmnet(x, y, family='binomial', alpha=0, parallel=TRUE, standardize=TRUE)

# weights = 1/absolute value of ridge coefficients
w3 <- 1/abs(matrix(coef(cv.ridge, s=cv.ridge$lambda.min)
                   [, 1][2:(ncol(x)+1)] ))^1 ## Using gamma = 1
w3[w3[,1] == Inf] <- 999999999 ## Replacing values estimated as Infinite for 999999999

# adaptive Lasso
set.seed(123)
cv.lasso <- cv.glmnet(x, y, family='binomial', alpha=1, parallel=TRUE, standardize=TRUE, type.measure='auc', penalty.factor=w3)

#####
# evaluation
#####

#based on Lessmann et al. 3 performance measures

#AUC
xpreds <- gold
prob=predict(cv.lasso, newx = x,type=c("response"))
xpreds$prob=prob
library(pROC)
g <- roc(good ~ prob, data = xpreds)
plot(g)    
AUC <- g$auc
#0.7973


#PG

#𝐺𝑖𝑛𝑖=2*partial 𝐴𝑈𝐶/(a+b)(b-a) − 1

(2*auc(good ~ prob, data = as.data.frame(xpreds), partial.auc = c(0,0.5))/((0+0.5)*(0.5-0)))-1
2*(AUC-0.5)
#BS
#accuracy: closer to 0 = better (1/N)*sum((f-o)²)

(BS <- sum((prob-y)^2)/nrow(y))
#0.1617

# plots
plot(cv.lasso)
plot(cv.lasso$glmnet.fit, xvar="lambda", label=TRUE)
abline(v = log(cv.lasso$lambda.min))
abline(v = log(cv.lasso$lambda.1se))
coef(cv.lasso, s=cv.lasso$lambda.1se)
coef <- coef(cv.lasso, s='lambda.1se')
selected_attributes <- (coef@i[-1]+1) ## Considering the structure of the data frame dataF as shown earlier

#####Rule ensembles

#adaptive lasso

#####SRE

#adaptive lasso