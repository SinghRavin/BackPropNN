#' The title of -sigmoid-
#'
#' Here is a brief description
#'
#' @param a Numeric matrix.
#'
#' @details Computes the sigmoid function value of each element of matrix \code{a}.
#' @return A matrix of class \code{BackPropNN_sigmoid}:
#' @examples
#' mat <- matrix(1:9,3,3)
#' sigmoid(mat)
#'
#' @export
sigmoid <- function(a){
  ans <- 1/(1+exp(-a))
  structure(list(a=a, value=ans), class = "BackPropNN_sigmoid")
}
