
#' The title of -feed_forward-
#'
#' Here is a brief description
#'
#' @param data R data frame with X columns and Y labels.
#' @param nn_model A list of weight and bias matrices along with other information about fitted nn model.
#'
#' @details Computes the predicted value of the X columns of \code{data}.
#' @return A vector of class \code{BackPropNN_feed_forward}:
#' @examples
#' data <- data.frame(X1 = 1:10, X2 = 2:11, Y = sample(c(0,1), 10, replace=TRUE))
#' nn_model <- back_propagation_training(i, h, o, learning_rate, activation_func, data)
#' feed_forward(data, nn_model)
#'
#' @export
feed_forward <- function(data, nn_model){
  X = as.matrix(data[1:ncol(data)-1])

  if (nn_model$activation_function=="sigmoid"){
    activ_func <- sigmoid
  } else if (nn_model$activation_function=="ReLU"){
    activ_func <- ReLU
  } else {
    stop("The activation function specified is not excepted. Choose either sigmoid or ReLU.")
  }

  weight_bias_matrices_list <- nn_model$weight_bias_matrices

  guess = numeric(nrow(X))
  for (i in 1:nrow(X)){
    hidden_matrix = map(\(a) activ_func(a)$value,
                        (weight_bias_matrices_list[[1]] %*% X[i,]) +
      weight_bias_matrices_list[[3]])$mapped_matrix
    guess[i] = map(\(a) activ_func(a)$value,
                   (weight_bias_matrices_list[[2]] %*% hidden_matrix) +
                     weight_bias_matrices_list[[4]])$mapped_matrix
  }
  structure(list(pred=guess), class = "BackPropNN_feed_forward")
}
