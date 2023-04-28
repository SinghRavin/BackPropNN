#include "my_header.h"
#include <RcppArmadillo.h>
using namespace Rcpp;
// [[Rcpp::depends(RcppArmadillo)]]

//' The title of -sigmoid_rcpp-
//'
//' Here is a brief description
//'
//' @param a Numeric matrix.
//'
//' @details Computes the sigmoid function value of each element of matrix \code{a}.
//' @return A matrix.
//' @examples
//' mat <- matrix(1:9,3,3)
//' sigmoid_rcpp(mat)
//'
//' @export
//' @useDynLib BackPropNN, .registration=TRUE
//' @importFrom Rcpp sourceCpp
// [[Rcpp::export]]
arma::mat sigmoid_rcpp(const arma::mat a) {

  return 1.0/(1.0 + arma::exp(-a));

}

