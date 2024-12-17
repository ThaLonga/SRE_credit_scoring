# Data complexity analysis
if (!require("pacman")) install.packages("pacman") ; require("pacman")
p_load(ECoL)

#linearity
source("./src/misc.R")
source("./src/data_loader.R")
datasets <- load_data()
