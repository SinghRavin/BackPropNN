## Fair benchmark: equal number of passes over the data for all three engines.
## nnet runs maxit BFGS iterations; BackPropNN runs `epochs` SGD passes.
## The R and C++ engines must produce identical weights given the same seed.

library(BackPropNN)

set.seed(100)
num_obs <- 10000
X1   <- runif(num_obs, 18, 60)
X2   <- runif(num_obs, 100, 250)
prob <- plogis(-13.6 + 0.10 * X1 + 0.05 * X2)
Y    <- rbinom(num_obs, 1, prob)

data <- data.frame(X1 = as.numeric(scale(X1)),
                   X2 = as.numeric(scale(X2)),
                   Y  = Y)
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
set.seed(1); m_r    <- back_propagation_training(i, h, o, lr, af, data, epochs = E)
set.seed(1); m_cpp  <- back_propagation_training_rcpp(i, h, o, lr, af, as.matrix(data), epochs = E)
set.seed(1); m_nnet <- nnet::nnet(X, Ym, size = h, maxit = E, trace = FALSE)

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
## Averaged over 10 random initializations. Two framings:
##   equal passes over the data (matches the speed benchmark), and
##   each engine run to convergence.
## Oracle = AUC attainable from the true probabilities, i.e. the ceiling.
set.seed(100)
train_idx <- sample(seq_len(num_obs), size = round(0.7 * num_obs))
train <- data[train_idx, ]
test  <- data[-train_idx, ]

X_tr <- as.matrix(train[, 1:2]); Y_tr <- as.matrix(train[, 3])
X_te <- as.matrix(test[, 1:2]);  Y_te <- test$Y
prob_te <- prob[-train_idx]

score <- function(p) c(auc = as.numeric(pROC::auc(pROC::roc(Y_te, p, quiet = TRUE))),
                       mse = mean((Y_te - p)^2))

n_rep <- 10
out <- lapply(seq_len(n_rep), function(s) {
  set.seed(s); f_r    <- back_propagation_training(i, h, o, lr, af, train, epochs = E)
  set.seed(s); f_cpp  <- back_propagation_training_rcpp(i, h, o, lr, af, as.matrix(train), epochs = E)
  set.seed(s); f_nnE  <- nnet::nnet(X_tr, Y_tr, size = h, maxit = E,   trace = FALSE)
  set.seed(s); f_nnC  <- nnet::nnet(X_tr, Y_tr, size = h, maxit = 100, trace = FALSE)
  set.seed(s); f_r100 <- back_propagation_training(i, h, o, lr, af, train, epochs = 100)

  rbind(
    "BackPropNN, 10 epochs"  = score(feed_forward(test, f_r)$pred),
    "Rcpp, 10 epochs"        = score(as.numeric(feed_forward_rcpp(test, f_cpp))),
    "nnet, 10 iterations"    = score(as.numeric(stats::predict(f_nnE, X_te, type = "raw"))),
    "BackPropNN, 100 epochs" = score(feed_forward(test, f_r100)$pred),
    "nnet, 100 iterations"   = score(as.numeric(stats::predict(f_nnC, X_te, type = "raw")))
  )
})

accuracy <- as.data.frame(Reduce(`+`, out) / n_rep)
accuracy <- rbind("Oracle (true probabilities)" = score(prob_te), accuracy)
accuracy$engine <- rownames(accuracy)
accuracy <- accuracy[, c("engine", "auc", "mse")]
print(accuracy, row.names = FALSE, digits = 4)
