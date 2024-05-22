#code from https://github.com/b0rxa/scmamp/blob/master/R/post_hoc.R

# NON-EXPORTED, AUXILIAR FUNCTIONS --------------------------------------------

processControlColumn <- function(data, control){
  # Function to process the control argument used in some functions in this script
  #
  # Args:
  #   data:    Dataset where the column is
  #   control: Name or ID of the column
  #
  # Returns:
  #   The ID of the column. If there is any problem, an error is rised.
  #
  if(!is.null(control)) {
    if (is.character(control)) {
      control <- which(names(data) %in% control)
      if (length(control)==0) {
        stop("The name of the column to be used as control does not exist ",
             "in the data matrix")
      }
    } else {
      if (control > ncol(data) | control < 1) {
        stop("Non-valid value for the control parameter. It has to be either ",
             "the name of a column or a number between 1 and ", ncol(data))
      }
    }
  }
  return(control)
}

generatePairs <- function(k, control) {
  # Non-exported function to create the pairs needed for the comparisons
  #
  # Args:
  #  k:       Number of algorithms.
  #  control: Id of the control. It can be NULL to indicate that all pairs have
  #           to be created.
  #
  # Returns:
  #   A data.frame with all the pairs to be compared
  #
  
  if(is.null(control)) { 
    # All vs all comparison
    aux.pairs <- sapply(1:(k - 1), 
                        FUN=function(x) {
                          cbind((x),(x+1):k)
                        })
    pairs <- do.call(what=rbind, args=aux.pairs)
  }  else{ 
    # All vs control comparison
    pairs <- cbind(control, (1:k)[-control])
  }
  return(pairs)
}

buildPvalMatrix <- function(pvalues, k, pairs, cnames, control){
  # Function to create the final p-value matrix
  #
  # Args:
  #   pvalues: Vector of p-values.
  #   k:       Number of algorithms.
  #   pairs:   Pair of algorithms to which each p-value correspond.
  #   cnames:  Character vector with the names of the algorithms.
  #   control: Id of the algorithm used as control (if any). Only usedt to know
  #            the type of comparison.
  #
  # Returns:
  #   A correctly formated matrix with the results.
  #
  if(is.null(control)){ 
    # All vs all comparison
    matrix.raw <- matrix(rep(NA, times=k^2), nrow=k)
    # The matrix has to be symetric
    matrix.raw[pairs]            <- pvalues
    matrix.raw[pairs[, c(2, 1)]] <- pvalues
    colnames(matrix.raw) <- cnames
    rownames(matrix.raw) <- cnames
  } else { 
    # All vs control comparison
    matrix.raw <- matrix(rep(NA, k), ncol=k)
    matrix.raw[(1:k)[-control]] <- pvalues
    colnames(matrix.raw) <- cnames    
  }  
  return(matrix.raw)
}


correctForMonotocity <- function (pvalues){
  # Function to ensure the monotocity of the p-values after being corrected
  # Args:
  #   pvalues: Corrected p-values, in DECRESING ORDER ACCORDING TO THE RAW PVALUE
  # Returns:
  #   The corrected p-values such as there is not a p-value smaller than the any 
  #   of the previous ones.
  #
  pvalues <- sapply(1:length(pvalues), 
                    FUN=function(x) {
                      return(max(pvalues[1:x]))
                    })
  return(pvalues)
}

setUpperBound <- function (vector, bound){
  # Simple function to limit the maximum value of a vector to a given value
  #
  # Args:
  #   vector: Vector whose values will be bounded
  #   bound:  Maximum value in the result
  #
  # Returns:
  #   A vector equal to 'vector' except for those values above 'bound', that are
  #   set to this upper bound
  #
  res <- sapply(vector, 
                FUN=function(x) {
                  return(min(bound, x))
                })
  return(res)
}

#' @title Rom correction of p-values
#'
#' @description This function takes the particular list of possible hypthesis to correct for multiple testing, as defined in Rom (1990)
#' @param pvalues Raw p-values in either a vector or a matrix. Note that in case the p-values are in a matrix, all the values are used for the correction. Therefore, if the matrix contains repeated values (as those output by some methods in this package), the repetitions have to be removed.
#' @param alpha value for the averall test
#' @return A vector or matrix with the corrected p-values
#' @details The test has been implemented according to the version in Garcia\emph{et al.} (2010), page 2680-2682.
#' @references S. Garcia, A. Fernandez, J. Luengo and F. Herrera (2010) Advanced nonparametric tests for multiple comparison in the design of experiments in computational intelligence and data mining: Experimental analysis of power. \emph{Information Sciences}, 180, 2044-2064.
#' @references D. M. Rom (1990) A sequentially rejective test procedure based on a modified Bonferroni inequality. \emph{Biometrika}, 77, 663-665.
#' @examples
#' data(data_gh_2008)
#' raw.pvalues <- friedmanPost(data.gh.2008)
#' raw.pvalues
#' adjustRom(raw.pvalues, alpha=0.05)
adjustRom <- function(pvalues, alpha=0.05){
  ord <- order(pvalues, na.last=NA)
  pvalues.sorted <- pvalues[ord]
  k <- length(pvalues.sorted) + 1  
  
  alpha.adj <- numeric(k - 1)
  alpha.adj[k - 1] <- alpha
  alpha.adj[k - 2] <- alpha/2  
  for(i in 3:(k - 1)){
    j1 <- 1:(i - 1)
    j2 <- 1:(i - 2)
    aux <- choose(i,j2) * (alpha.adj[k - 1 - j2])^(i - j2)
    alpha.adj[k-i] <- (sum(alpha^j1) - sum(aux)) / i      
  }
  
  r <- alpha / alpha.adj
  
  p.val.aux <- r * pvalues.sorted
  p.val.aux <- setUpperBound(p.val.aux, 1)
  p.adj.aux <- correctForMonotocity(pvalues=p.val.aux)
  
  p.adj.aux <- sapply((k - 1):1,
                      FUN=function(i) {
                        min(p.adj.aux[(k-1):i])
                      })
  
  # Note that the code above ensures that any value is not bigger than any of the
  # values bellow, but it reverse the order of the corrected p-values.
  
  p.adj.aux <- rev(p.adj.aux)
  
  p.adj <- rep(NA,length(pvalues))
  suppressWarnings(expr = {
    p.adj[ord] <- p.adj.aux
  })
  
  if(is.matrix(pvalues)){
    p.adj <- matrix(p.adj, ncol=ncol(pvalues))
    colnames(p.adj) <- colnames(pvalues)
    rownames(p.adj) <- rownames(pvalues)
  }
  return(p.adj)
}