#' The title of -derivative_sigmoid-
#'
#' Here is a brief description
#'
#' @param a Numeric matrix.
#'
#' @details Computes the derivative of sigmoid function value of each element of matrix \code{a}.
#' @return A matrix of class \code{BackPropNN_derivative_sigmoid}:
#' @examples
#' mat <- matrix(1:9,3,3)
#' derivative_sigmoid(mat)
#'
#' @export
derivative_sigmoid <- function(a){
  ans <- as.matrix(a*(1-a))
  structure(list(a=a, value=ans), class = "BackPropNN_derivative_sigmoid")
}
