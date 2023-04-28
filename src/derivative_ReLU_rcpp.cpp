#include "my_header.h"
#include <RcppArmadillo.h>
using namespace Rcpp;
// [[Rcpp::depends(RcppArmadillo)]]

//' The title of -derivative_ReLU_rcpp-
//'
//' Here is a brief description
//'
//' @param a Numeric matrix.
//'
//' @details Computes the derivative of ReLU function value of each element of matrix \code{a}.
//' @return A matrix.
//' @examples
//' mat <- matrix(1:9,3,3)
//' derivative_ReLU_rcpp(mat)
//'
//' @export
// [[Rcpp::export]]
arma::mat derivative_ReLU_rcpp(arma::mat a) {

  return arma::conv_to<arma::mat>::from(a > 0);

}
