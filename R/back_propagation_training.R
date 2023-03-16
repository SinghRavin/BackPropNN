#' The title of -back_propagation_training-
#'
#' Here is a brief description
#'
#' @param i Numeric scalar. Number of input nodes.
#' @param h Numeric scalar. Number of hidden nodes.
#' @param o Numeric scalar. Number of output nodes.
#' @param learning_rate Numeric scalar. Learning rate of the algorithm.
#' @param activation_func Character (either "sigmoid" or "ReLU").
#' @param data R data frame with X columns and Y labels.
#'
#' @details Computes the weight and bias matrices for the nodes of the neural network with \code{i} # of input nodes, \code{h} # of hidden nodes, and \code{o} # of output nodes.
#' @return A list of class \code{BackPropNN_back_propagation_training}:
#' @examples
#' set.seed(100)
#' data <- data.frame(X1 = 1:100, X2 = 2:101, Y = sample(c(0,1), 100, replace=TRUE))
#' nn_model <- back_propagation_training(i=2, h=2, o=1, learning_rate=0.01,
#' activation_func="sigmoid", data=data)
#'
#' @export
back_propagation_training <- function(i, h, o, learning_rate, activation_func, data){

  W_IH = matrix(0.01,nrow=h,ncol=i)
  W_HO = matrix(0.01,nrow=o,ncol=h)
  B_H = matrix(0.01,nrow=h,ncol=1)
  B_O = matrix(0.01,nrow=o,ncol=1)

  X = as.matrix(data[1:ncol(data)-1])
  Y = as.matrix(data[,ncol(data)])

  if (activation_func=="sigmoid"){
    activ_func <- sigmoid
    activ_func_deriv <- derivative_sigmoid
  } else if (activation_func=="ReLU"){
    activ_func <- ReLU
    activ_func_deriv <- derivative_ReLU
  } else {
    stop("The activation function specified is not excepted. Choose either sigmoid or ReLU.")
  }

  for (j in 1:nrow(X)){
    input_matrix_transpose = t(X[j,])
    hidden_matrix = activ_func((W_IH %*% as.matrix(X[j,])) + B_H)$value
    hidden_matrix_transpose = t(hidden_matrix)

    output_matrix = activ_func((W_HO %*% hidden_matrix) + B_O)$value
    output_error = as.matrix(Y[j,]) - output_matrix

    hidden_error = t(W_HO) %*% output_error

    gradient_HO = learning_rate*(output_error*activ_func_deriv(output_matrix)$value)
    Weight_HO_deltas = gradient_HO %*% hidden_matrix_transpose

    W_HO = W_HO + Weight_HO_deltas
    B_O = B_O + gradient_HO

    gradient_IH = learning_rate*(hidden_error*activ_func_deriv(hidden_matrix)$value)
    Weight_IH_deltas = gradient_IH %*% input_matrix_transpose

    W_IH = W_IH + Weight_IH_deltas
    B_H = B_H + gradient_IH
  }

  structure(list(input_data=data, num_nodes=stats::setNames(c(i,h,o), c("# of input nodes",
                                                                          "# of hidden nodes",
                                                                          "# of output nodes")),
                 activation_function=activation_func,
                 learning_rate = learning_rate,
                 weight_bias_matrices = list(weight_input_hidden=W_IH,
                                             weight_hidden_output=W_HO,
                                             bias_hidden=B_H,
                                             bias_output=B_O)), class = "BackPropNN_back_propagation_training")
}

#' @rdname back_propagation_training
#' @export
#' @param x An object of class \code{BackPropNN_back_propagation_training}.
plot.BackPropNN_back_propagation_training <- function(x) {

  data <- x$input_data
  X = as.matrix(data[1:ncol(data)-1])
  Y = as.matrix(data[,ncol(data)])

  nn_R <- nnet::nnet(X,Y,size=x$num_nodes[2], trace=FALSE)
  nn_R_pred <- as.numeric(stats::predict(nn_R,X, type="raw"))

  pROC::roc(data[,ncol(data)],feed_forward(data,x)$pred,
            plot=TRUE, print.auc=TRUE, main="ROC curve by BackPropNN")
  pROC::roc(data[,ncol(data)],nn_R_pred,
            plot=TRUE, print.auc=TRUE, main="ROC curve by R nnet")
}

#' @rdname back_propagation_training
#' @export
#' @param x An object of class \code{BackPropNN_back_propagation_training}.
summary.BackPropNN_back_propagation_training <- function(x) {
  list(num_nodes=x$num_nodes,
       activation_function=x$activation_function,
       learning_rate = x$learning_rate,
       weight_bias_matrices = x$weight_bias_matrices)
}

#' @rdname back_propagation_training
#' @export
#' @param x An object of class \code{BackPropNN_back_propagation_training}.
print.BackPropNN_back_propagation_training <- function(x) {

  data <- x$input_data
  X = as.matrix(data[1:ncol(data)-1])
  Y = as.matrix(data[,ncol(data)])

  nn_R <- nnet::nnet(X,Y,size=x$num_nodes[2], trace=FALSE)
  nn_R_pred <- as.numeric(stats::predict(nn_R,X, type="raw"))
  nn_R_mse <- mean((Y - nn_R_pred)^2)

  my_nn_mse <- mean((Y - feed_forward(x$input_data,x)$pred)^2)

  print(bench::mark("BackPropNN"=back_propagation_training(x$num_nodes[1],x$num_nodes[2],
                                          x$num_nodes[3], x$learning_rate,
                                          x$activation_function, data),
                    "R nnet"=nnet::nnet(X,Y,size=x$num_nodes[2], trace=FALSE), relative = TRUE, check = FALSE))

  list(mse_comparison =stats::setNames(c(nn_R_mse,my_nn_mse),
                                       c("MSE by R nnet", "MSE by BackPropNN")))
}





