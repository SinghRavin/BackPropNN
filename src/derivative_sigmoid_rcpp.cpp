#include "my_header.h"
#include <RcppArmadillo.h>
using namespace Rcpp;
// [[Rcpp::depends(RcppArmadillo)]]

//' The title of -derivative_sigmoid_rcpp-
//'
//' Here is a brief description
//'
//' @param a Numeric matrix.
//'
//' @details Computes the derivative of sigmoid function value of each element of matrix \code{a}.
//' @return A matrix.
//' @examples
//' mat <- matrix(1:9,3,3)
//' derivative_sigmoid_rcpp(mat)
//'
//' @export
// [[Rcpp::export]]
arma::mat derivative_sigmoid_rcpp(const arma::mat a) {

   arma::mat Y = arma::ones<arma::mat>(a.n_rows, a.n_cols);

   return a%(Y-a);

 }
