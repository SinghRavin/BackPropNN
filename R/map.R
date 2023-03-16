#' The title of -map-
#'
#' Here is a brief description
#'
#' @param a Numeric matrix.
#' @param func R function.
#'
#' @details Computes the function value (func) of each element of matrix \code{a}.
#' @return A matrix (of same dimension as matrix a) of class \code{BackPropNN_map}:
#' @examples
#' mat <- matrix(1:9,3,3)
#' func <- function(x){
#' return(x+2)
#' }
#' map(func, mat)
#'
#' @export
map <- function(func, a){
  ans <- as.matrix(apply(a, 2, func))
  structure(list(a=a, mapped_matrix=ans), class = "BackPropNN_map")
}
