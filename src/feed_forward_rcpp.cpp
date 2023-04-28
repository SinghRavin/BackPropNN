#include "my_header.h"
#include <RcppArmadillo.h>
using namespace Rcpp;
// [[Rcpp::depends(RcppArmadillo)]]

//' The title of -feed_forward_rcpp-
//'
//' Here is a brief description
//'
//' @param data R data frame with X columns and Y labels.
//' @param nn_model A list of weight and bias matrices along with other information about fitted nn model.
//'
//' @details Computes the predicted value of the X columns of \code{data}.
//' @return A vector.
//' @examples
//' data <- data.frame(X1 = 1:10, X2 = 2:11, Y = sample(c(0,1), 10, replace=TRUE))
//' data_as_matrix <- as.matrix(data)
//' nn_model_rcpp <- back_propagation_training_rcpp(i=2, h=2, o=1, learning_rate=0.01,
//' activation_func="sigmoid", data=data_as_matrix)
//' feed_forward_rcpp(data, nn_model_rcpp)
//'
//' @export
// [[Rcpp::export]]
arma::vec feed_forward_rcpp(Rcpp::DataFrame data, List nn_model) {

  // Get input matrix
  int n_rows = data.nrows();
  int n_cols = data.size() - 1; // drop last column
  arma::mat X(n_rows, n_cols);

  for (int i = 0; i < n_cols; i++) {
    X.col(i) = Rcpp::as<arma::vec>(data[i]);
  }

  // Setting up activation function
  std::string activation_function = as<std::string>(nn_model["activation_function"]);
  arma::mat (*activ_func)(arma::mat);

  if (activation_function == "sigmoid") {
    activ_func = &sigmoid_rcpp;
  } else if (activation_function == "ReLU") {
    activ_func = &ReLU_rcpp;
  } else {
    throw std::invalid_argument("The activation function specified is not accepted. Choose either sigmoid or ReLU.");
  }

  // Get weight and bias matrices
  List weight_bias_matrices_list = nn_model["weight_bias_matrices"];
  arma::mat W1 = weight_bias_matrices_list[0];
  arma::mat W2 = weight_bias_matrices_list[1];
  arma::vec b1 = weight_bias_matrices_list[2];
  arma::vec b2 = weight_bias_matrices_list[3];

  // Perform feed forward calculation for each row of X
  arma::vec guess(X.n_rows);
  for (int i = 0; i < static_cast<int>(X.n_rows); i++) {
    arma::mat hidden_matrix = (*activ_func)(W1 * X.row(i).t() + b1);
    guess(i) = as_scalar((*activ_func)(W2 * hidden_matrix + b2));
  }

  return guess;
}
