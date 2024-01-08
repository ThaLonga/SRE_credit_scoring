load_data <- function () {
  for(dataset in c("german", "australian", "kaggle")) {
    load(paste("data/GOLD/",dataset,".Rda", sep=""))
  }
  return(list(german_dummies, australian_dummies, kaggle_imputed))
}