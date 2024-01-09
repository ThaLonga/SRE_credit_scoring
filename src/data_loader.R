load_data <- function () {
  for(dataset in c("german", "australian", "kaggle")) {
    load(paste("data/GOLD/",dataset,".Rda", sep=""))
  }
  return(list(german, australian, kaggle, thomas))
}