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
#' @param epochs Numeric scalar. Number of complete passes over the data.
#' @param ... Further arguments passed.
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
back_propagation_training <- function(i, h, o, learning_rate,
                                      activation_func, data, epochs = 1, ...){

  W_IH = matrix(stats::rnorm(h*i, 0, sqrt(1/i)), nrow=h, ncol=i)
  W_HO = matrix(stats::rnorm(o*h, 0, sqrt(1/h)), nrow=o, ncol=h)
  B_H = matrix(0, nrow=h, ncol=1)
  B_O = matrix(0, nrow=o, ncol=1)

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

  for (e in seq_len(epochs)){
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
}
  structure(list(input_data=data, num_nodes=stats::setNames(c(i,h,o), c("# of input nodes",
                                                                          "# of hidden nodes",
                                                                          "# of output nodes")),
                 activation_function=activation_func,
                 learning_rate=learning_rate,
                 weight_bias_matrices=list(weight_input_hidden=W_IH,
                                             weight_hidden_output=W_HO,
                                             bias_hidden=B_H,
                                             bias_output=B_O)), class = "BackPropNN_back_propagation_training")
}

#' @rdname back_propagation_training
#' @export
#' @param x An object of class \code{BackPropNN_back_propagation_training}.
#' @param newdata Optional data frame on which to evaluate the model, with the
#'   outcome in the last column. Defaults to the training data, in which case
#'   the ROC curve is in-sample.
#' @param ... Further arguments passed.
plot.BackPropNN_back_propagation_training <- function(x, newdata = NULL, ...) {
  in_sample <- is.null(newdata)
  data <- if (in_sample) x$input_data else newdata
  y <- data[, ncol(data)]
  r <- pROC::roc(y, feed_forward(data, x)$pred, plot = TRUE, print.auc = TRUE,
                 quiet = TRUE,
                 main = if (in_sample) "ROC curve by BackPropNN (in-sample)"
                 else "ROC curve by BackPropNN (held-out)")
  invisible(r)
}

#' @rdname back_propagation_training
#' @export
#' @param object An object of class \code{BackPropNN_back_propagation_training}.
#' @param ... Further arguments passed.
summary.BackPropNN_back_propagation_training <- function(object, ...) {

  list(num_nodes=object$num_nodes,
       activation_function=object$activation_function,
       learning_rate=object$learning_rate,
       weight_bias_matrices=object$weight_bias_matrices)
}

#' @rdname back_propagation_training
#' @export
#' @param x An object of class \code{BackPropNN_back_propagation_training}.
#' @param ... Further arguments passed.
print.BackPropNN_back_propagation_training <- function(x, ...) {
  y <- x$input_data[, ncol(x$input_data)]
  cat("A 3-layer neural network trained by backpropagation\n\n")
  cat("  Input nodes   :", x$num_nodes[1], "\n")
  cat("  Hidden nodes  :", x$num_nodes[2], "\n")
  cat("  Output nodes  :", x$num_nodes[3], "\n")
  cat("  Activation    :", x$activation_function, "\n")
  cat("  Learning rate :", x$learning_rate, "\n")
  cat("  Training rows :", nrow(x$input_data), "\n")
  cat("  In-sample MSE :",
      round(mean((y - feed_forward(x$input_data, x)$pred)^2), 4), "\n")
  invisible(x)
}




