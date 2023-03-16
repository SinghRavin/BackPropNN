#' The title of -ReLU-
#'
#' Here is a brief description
#'
#' @param a Numeric matrix.
#'
#' @details Computes the ReLU function value of each element of matrix \code{a}.
#' @return A matrix of class \code{BackPropNN_ReLU}:
#' @examples
#' mat <- matrix(1:9,3,3)
#' ReLU(mat)
#'
#' @export
ReLU <- function(a){
  ans = Rmpfr::pmax(a,0)
  structure(list(a=a, value=ans), class = "BackPropNN_ReLU")
}
