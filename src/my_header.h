#ifndef MY_HEADER_H
#define MY_HEADER_H

#include <RcppArmadillo.h>

arma::mat sigmoid_rcpp(arma::mat X);
arma::mat ReLU_rcpp(arma::mat X);
arma::mat derivative_sigmoid_rcpp(arma::mat X);
arma::mat derivative_ReLU_rcpp(arma::mat X);

#endif
