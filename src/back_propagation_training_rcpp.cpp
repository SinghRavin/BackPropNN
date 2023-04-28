#include "my_header.h"
#include <RcppArmadillo.h>
using namespace Rcpp;
// [[Rcpp::depends(RcppArmadillo)]]

//' The title of -back_propagation_training_rcpp-
//'
//' Here is a brief description
//'
//' @param i Numeric scalar. Number of input nodes.
//' @param h Numeric scalar. Number of hidden nodes.
//' @param o Numeric scalar. Number of output nodes.
//' @param learning_rate Numeric scalar. Learning rate of the algorithm.
//' @param activation_func Character (either "sigmoid" or "ReLU").
//' @param data R data frame with X columns and Y labels.
//'
//' @details Computes the weight and bias matrices for the nodes of the neural network with \code{i} # of input nodes, \code{h} # of hidden nodes, and \code{o} # of output nodes.
//' @return A list of weight and bias matrices.
//' @examples
//' set.seed(100)
//' data <- data.frame(X1 = 1:100, X2 = 2:101, Y = sample(c(0,1), 100, replace=TRUE))
//' data_as_matrix <- as.matrix(data)
//' nn_model_rcpp <- back_propagation_training_rcpp(i=2, h=2, o=1, learning_rate=0.01,
//' activation_func="sigmoid", data=data_as_matrix)
//'
//' @export
// [[Rcpp::export]]
List back_propagation_training_rcpp(int i, int h, int o, double learning_rate,
                                    std::string activation_func, arma::mat data){

  arma::mat W_IH(h, i, arma::fill::ones);
  W_IH *= 0.01; // Initializing the weights to be 0.01
  arma::mat W_HO(o, h, arma::fill::ones);
  W_HO *= 0.01;
  arma::mat B_H(h, 1, arma::fill::ones);
  B_H *= 0.01;
  arma::mat B_O(o, 1, arma::fill::ones);
  B_O *= 0.01;

  // Extracting all but last columns into a new matrix X
  arma::mat X = data.head_cols(data.n_cols - 1);

  // Extract the last column into a new matrix Y
  arma::mat Y = data.tail_cols(1);

  arma::mat (*activ_func)(arma::mat);
  arma::mat (*activ_func_deriv)(arma::mat);

  if (activation_func=="sigmoid"){
    activ_func = &sigmoid_rcpp;
    activ_func_deriv = &derivative_sigmoid_rcpp;
  } else if (activation_func=="ReLU"){
    activ_func = &ReLU_rcpp;
    activ_func_deriv = &derivative_ReLU_rcpp;
  } else {
    throw std::invalid_argument("The activation function specified is not accepted. Choose either sigmoid or ReLU.");
  }

  for (int j = 0; j < static_cast<int>(X.n_rows); j++){

    arma::mat row_X = arma::reshape(X.row(j), X.n_cols, 1);
    arma::mat row_Y = arma::reshape(Y.row(j), Y.n_cols, 1);

    arma::mat input_matrix_transpose = row_X.t();
    arma::mat hidden_matrix = (*activ_func)(W_IH * row_X + B_H);
    arma::mat hidden_matrix_transpose = hidden_matrix.t();

    arma::mat output_matrix = (*activ_func)(W_HO * hidden_matrix + B_O);
    arma::mat output_error = row_Y - output_matrix;

    arma::mat hidden_error = W_HO.t() * output_error;

    arma::mat gradient_HO = learning_rate * (output_error % (*activ_func_deriv)(output_matrix));
    arma::mat Weight_HO_deltas = gradient_HO * hidden_matrix_transpose;

    W_HO = W_HO + Weight_HO_deltas;
    B_O = B_O + gradient_HO;

    arma::mat gradient_IH = learning_rate * (hidden_error % (*activ_func_deriv)(hidden_matrix));
    arma::mat Weight_IH_deltas = gradient_IH * input_matrix_transpose;

    W_IH = W_IH + Weight_IH_deltas;
    B_H = B_H + gradient_IH;
  }

  // arma::mat nn_model = arma::join_vert(arma::vectorise(W_IH), arma::join_vert(arma::vectorise(W_HO), arma::join_vert(B_H, B_O)));

  arma::ivec num_nodes = {i, h, o};

  return List::create(Named("input_data") = data,
                      Named("num_nodes") = num_nodes,
                      Named("activation_function") = activation_func,
                      Named("learning_rate") = learning_rate,
                      Named("weight_bias_matrices") = List::create(Named("weight_input_hidden") = W_IH,
                            Named("weight_hidden_output") = W_HO,
                            Named("bias_hidden") = B_H,
                            Named("bias_output") = B_O));
}
