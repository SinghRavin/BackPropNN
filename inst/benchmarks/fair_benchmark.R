## Fair benchmark: equal number of passes over the data for all three engines.
## nnet runs maxit BFGS iterations; BackPropNN runs `epochs` SGD passes.

library(BackPropNN)

set.seed(100)
num_obs <- 10000
X1   <- runif(num_obs, 18, 60)
X2   <- runif(num_obs, 100, 250)
prob <- plogis(-2.5 + 0.02 * X1 + 0.01 * X2)
Y    <- rbinom(num_obs, 1, prob)

data <- data.frame(X1, X2, Y)
X    <- as.matrix(data[, 1:2])
Ym   <- as.matrix(data[, 3])

i <- 2; h <- 4; o <- 1; lr <- 0.1; af <- "sigmoid"
E <- 10   # passes over the data, held equal across engines

## ---- Training -------------------------------------------------------------
set.seed(100)
res_train <- bench::mark(
  "Pure R" = back_propagation_training(i, h, o, lr, af, data, epochs = E),
  "Rcpp"   = back_propagation_training_rcpp(i, h, o, lr, af, as.matrix(data), epochs = E),
  "nnet"   = nnet::nnet(X, Ym, size = h, maxit = E, trace = FALSE),
  check = FALSE, iterations = 20
)
print(res_train)
print(summary(res_train, relative = TRUE))

## ---- Prediction -----------------------------------------------------------
m_r    <- back_propagation_training(i, h, o, lr, af, data, epochs = E)
m_cpp  <- back_propagation_training_rcpp(i, h, o, lr, af, as.matrix(data), epochs = E)
m_nnet <- nnet::nnet(X, Ym, size = h, maxit = E, trace = FALSE)

res_pred <- bench::mark(
  "Pure R" = feed_forward(data, m_r),
  "Rcpp"   = feed_forward_rcpp(data, m_cpp),
  "nnet"   = as.numeric(stats::predict(m_nnet, X, type = "raw")),
  check = FALSE, iterations = 100
)
print(res_pred)
print(summary(res_pred, relative = TRUE))

## ---- Equivalence check ----------------------------------------------------
## The R and C++ engines must produce identical weights.
stopifnot(isTRUE(all.equal(
  unname(m_r$weight_bias_matrices$weight_input_hidden),
  unname(m_cpp$weight_bias_matrices$weight_input_hidden)
)))

## ---- Accuracy on held-out data --------------------------------------------
## Trained on 70%, evaluated on the unseen 30%. Same epochs for all engines.
set.seed(100)
train_idx <- sample(seq_len(num_obs), size = round(0.7 * num_obs))
train <- data[train_idx, ]
test  <- data[-train_idx, ]

X_tr <- as.matrix(train[, 1:2]); Y_tr <- as.matrix(train[, 3])
X_te <- as.matrix(test[, 1:2]);  Y_te <- test$Y

fit_r    <- back_propagation_training(i, h, o, lr, af, train, epochs = E)
fit_cpp  <- back_propagation_training_rcpp(i, h, o, lr, af, as.matrix(train), epochs = E)
fit_nnet <- nnet::nnet(X_tr, Y_tr, size = h, maxit = E, trace = FALSE)

p_r    <- feed_forward(test, fit_r)$pred
p_cpp  <- as.numeric(feed_forward_rcpp(test, fit_cpp))
p_nnet <- as.numeric(stats::predict(fit_nnet, X_te, type = "raw"))

accuracy <- data.frame(
  engine = c("Pure R", "Rcpp", "nnet"),
  auc = c(as.numeric(pROC::auc(pROC::roc(Y_te, p_r,   quiet = TRUE))),
          as.numeric(pROC::auc(pROC::roc(Y_te, p_cpp,  quiet = TRUE))),
          as.numeric(pROC::auc(pROC::roc(Y_te, p_nnet, quiet = TRUE)))),
  mse = c(mean((Y_te - p_r)^2),
          mean((Y_te - p_cpp)^2),
          mean((Y_te - p_nnet)^2))
)
print(accuracy)
