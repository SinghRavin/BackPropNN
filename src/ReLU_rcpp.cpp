#include "my_header.h"
#include <RcppArmadillo.h>
using namespace Rcpp;
// [[Rcpp::depends(RcppArmadillo)]]

//' The title of -ReLU_rcpp-
//'
//' Here is a brief description
//'
//' @param a Numeric matrix.
//'
//' @details Computes the ReLU function value of each element of matrix \code{a}.
//' @return A matrix.
//' @examples
//' mat <- matrix(1:9,3,3)
//' ReLU_rcpp(mat)
//'
//' @export
// [[Rcpp::export]]

arma::mat ReLU_rcpp(const arma::mat a) {

  arma::mat Y = arma::zeros<arma::mat>(a.n_rows, a.n_cols);

  return arma::max(a, Y);

}
