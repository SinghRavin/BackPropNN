#' @importFrom stats setNames predict
NULL

#' @importFrom Rmpfr pmax
NULL

#' @importFrom pROC roc
NULL

#' @importFrom nnet nnet
NULL

#' @importFrom bench mark
NULL

#' BackPropNN
#'
#' A collection of functions
#'
#' @description This R-package will contain functions that will implement NN
#'  training (via back-propagation) from scratch (using basic R packages).
#'  The function will take data (X and Y) as an input and will ask user to specify
#'  no. of input nodes, no. of hidden nodes, no. of output nodes, learning rate, and an
#'  activation function to be used. The function will produce the NN model in terms
#'  of matrices containing weights for each nodes. The user will be able to choose
#'  activation function between ReLU and Sigmoid. The user will be able to compare
#'  the performance of this R-package in comparison to existing R NN packages in
#'  terms of accuracy and computational time.
#'
#' @docType package
#' @name BackPropNN
#'
#' @author Ravinder Singh
NULL
