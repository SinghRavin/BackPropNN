#' The title of -derivative_ReLU-
#'
#' Here is a brief description
#'
#' @param a Numeric matrix.
#'
#' @details Computes the derivative of ReLU function value of each element of matrix \code{a}.
#' @return A matrix of class \code{BackPropNN_derivative_ReLU}:
#' @examples
#' mat <- matrix(1:9,3,3)
#' derivative_ReLU(mat)
#'
#' @export
derivative_ReLU <- function(a){
  ans = as.matrix(ifelse(a<=0,0,1))
  structure(list(a=a, value=ans), class = "BackPropNN_derivative_ReLU")
}
