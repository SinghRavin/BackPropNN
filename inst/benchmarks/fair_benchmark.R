## fair_benchmark.R -----------------------------------------------------------
##
## Runs every benchmark and comparison reported in README.md and saves the
## results to inst/benchmarks/results.rds. The README reads that file rather
## than re-running the benchmarks, so the published numbers are stable across
## knits and come from one deliberate run on a quiet machine.
##
## To refresh the published numbers:
##   1. close other applications
##   2. restart R
##   3. source("inst/benchmarks/fair_benchmark.R")
##   4. devtools::build_readme()
##
## nnet optimises with BFGS (one weight update per iteration, using a gradient
## over the whole data set). BackPropNN uses stochastic gradient descent (one
## update per observation). The two are therefore compared under two separate
## protocols: speed at an equal number of passes over the data, and accuracy
## with each engine run to convergence under its own optimiser.

library(BackPropNN)

## ---- Data -----------------------------------------------------------------
set.seed(100)
num_obs <- 10000
beta0 <- -13.6
beta1 <- 0.10
beta2 <- 0.05

X1 <- runif(n = num_obs, min = 18,  max = 60)
X2 <- runif(n = num_obs, min = 100, max = 250)
prob <- exp(beta0 + beta1*X1 + beta2*X2) / (1 + exp(beta0 + beta1*X1 + beta2*X2))
Yvec <- rbinom(n = num_obs, size = 1, prob = prob)

## Predictors are standardised. A sigmoid saturates when inputs are on the raw
## scale (X2 averages 175), which stalls learning in every engine compared here.
data <- data.frame(X1 = as.numeric(scale(X1)),
                   X2 = as.numeric(scale(X2)),
                   Y  = Yvec)
X  <- as.matrix(data[, 1:2])
Ym <- as.matrix(data[, 3])

i <- 2; h <- 4; o <- 1
learning_rate <- 0.1
activation_func <- "sigmoid"
E <- 10   # passes over the data, held equal across engines for the speed test

## ---- Speed: training ------------------------------------------------------
res_train <- bench::mark(
  "Pure R" = back_propagation_training(i, h, o, learning_rate, activation_func,
                                       data, epochs = E),
  "Rcpp"   = back_propagation_training_rcpp(i, h, o, learning_rate,
                                            activation_func, as.matrix(data),
                                            epochs = E),
  "nnet"   = nnet::nnet(X, Ym, size = h, maxit = E, trace = FALSE),
  check = FALSE, iterations = 20
)

## ---- Speed: prediction ----------------------------------------------------
set.seed(1); m_r    <- back_propagation_training(i, h, o, learning_rate,
                                                 activation_func, data, epochs = E)
set.seed(1); m_cpp  <- back_propagation_training_rcpp(i, h, o, learning_rate,
                                                      activation_func,
                                                      as.matrix(data), epochs = E)
set.seed(1); m_nnet <- nnet::nnet(X, Ym, size = h, maxit = E, trace = FALSE)

res_pred <- bench::mark(
  "Pure R" = feed_forward(data, m_r),
  "Rcpp"   = feed_forward_rcpp(data, m_cpp),
  "nnet"   = as.numeric(stats::predict(m_nnet, X, type = "raw")),
  check = FALSE, iterations = 100
)

## ---- Equivalence ----------------------------------------------------------
## The R and C++ engines are the same algorithm; given the same seed they must
## produce the same weights.
engines_agree <- isTRUE(all.equal(
  unname(m_r$weight_bias_matrices$weight_input_hidden),
  unname(m_cpp$weight_bias_matrices$weight_input_hidden)
))
stopifnot(engines_agree)

## ---- Accuracy -------------------------------------------------------------
## Trained on 70%, evaluated on the held-out 30%, averaged over five random
## initialisations. The C++ engine is used for the repeated fits because it
## produces weights identical to the R engine at a fraction of the cost.
set.seed(100)
train_idx <- sample(seq_len(num_obs), size = round(0.7 * num_obs))
train <- data[train_idx, ]
test  <- data[-train_idx, ]

X_tr <- as.matrix(train[, 1:2]); Y_tr <- as.matrix(train[, 3])
X_te <- as.matrix(test[, 1:2]);  Y_te <- test$Y

score <- function(p) {
  c(auc = as.numeric(pROC::auc(pROC::roc(Y_te, p, quiet = TRUE))),
    mse = mean((Y_te - p)^2))
}

n_rep <- 5
out <- lapply(seq_len(n_rep), function(s) {
  set.seed(s); f10  <- back_propagation_training_rcpp(i, h, o, learning_rate, activation_func, as.matrix(train), epochs = 10)
  set.seed(s); f100 <- back_propagation_training_rcpp(i, h, o, learning_rate, activation_func, as.matrix(train), epochs = 100)
  set.seed(s); f500 <- back_propagation_training_rcpp(i, h, o, learning_rate, activation_func, as.matrix(train), epochs = 500)
  set.seed(s); f2000 <- back_propagation_training_rcpp(i, h, o, learning_rate, activation_func, as.matrix(train), epochs = 2000)
  set.seed(s); n10  <- nnet::nnet(X_tr, Y_tr, size = h, maxit = 10,  trace = FALSE)
  set.seed(s); n100 <- nnet::nnet(X_tr, Y_tr, size = h, maxit = 100, trace = FALSE)
  set.seed(s); n500 <- nnet::nnet(X_tr, Y_tr, size = h, maxit = 500, trace = FALSE)
  set.seed(s); n2000 <- nnet::nnet(X_tr, Y_tr, size = h, maxit = 2000, trace = FALSE)

  rbind(
    "BackPropNN, 10 epochs"  = score(as.numeric(feed_forward_rcpp(test, f10))),
    "BackPropNN, 100 epochs" = score(as.numeric(feed_forward_rcpp(test, f100))),
    "BackPropNN, 500 epochs" = score(as.numeric(feed_forward_rcpp(test, f500))),
    "BackPropNN, 2000 epochs" = score(as.numeric(feed_forward_rcpp(test, f2000))),
    "nnet, 10 iterations"    = score(as.numeric(stats::predict(n10,  X_te, type = "raw"))),
    "nnet, 100 iterations"   = score(as.numeric(stats::predict(n100, X_te, type = "raw"))),
    "nnet, 500 iterations"   = score(as.numeric(stats::predict(n500, X_te, type = "raw"))),
    "nnet, 2000 iterations"  = score(as.numeric(stats::predict(n2000, X_te, type = "raw")))
  )
})

accuracy <- as.data.frame(Reduce(`+`, out) / n_rep)
accuracy <- data.frame(engine = rownames(accuracy), accuracy, row.names = NULL)

## ---- Cost against accuracy ------------------------------------------------
## Timing is the median of five repetitions at a fixed seed. The function is
## passed in rather than its result, so each repetition is re-evaluated.
time_fit <- function(f, reps = 5) {
  stats::median(replicate(reps, system.time(f())[["elapsed"]]))
}

timings <- c(
  "Pure R, 10 epochs"    = time_fit(function() { set.seed(1); back_propagation_training(i, h, o, learning_rate, activation_func, train, epochs = 10) }),
  "Rcpp, 10 epochs"      = time_fit(function() { set.seed(1); back_propagation_training_rcpp(i, h, o, learning_rate, activation_func, as.matrix(train), epochs = 10) }),
  "Rcpp, 100 epochs"     = time_fit(function() { set.seed(1); back_propagation_training_rcpp(i, h, o, learning_rate, activation_func, as.matrix(train), epochs = 100) }),
  "Rcpp, 500 epochs"     = time_fit(function() { set.seed(1); back_propagation_training_rcpp(i, h, o, learning_rate, activation_func, as.matrix(train), epochs = 500) }),
  "Rcpp, 2000 epochs"     = time_fit(function() { set.seed(1); back_propagation_training_rcpp(i, h, o, learning_rate, activation_func, as.matrix(train), epochs = 2000) }),
  "nnet, 10 iterations"  = time_fit(function() { set.seed(1); nnet::nnet(X_tr, Y_tr, size = h, maxit = 10, trace = FALSE) }),
  "nnet, 100 iterations" = time_fit(function() { set.seed(1); nnet::nnet(X_tr, Y_tr, size = h, maxit = 100, trace = FALSE) }),
  "nnet, 500 iterations" = time_fit(function() { set.seed(1); nnet::nnet(X_tr, Y_tr, size = h, maxit = 500, trace = FALSE) }),
  "nnet, 2000 iterations" = time_fit(function() { set.seed(1); nnet::nnet(X_tr, Y_tr, size = h, maxit = 2000, trace = FALSE) })
)

acc_lookup <- c("Pure R, 10 epochs"    = "BackPropNN, 10 epochs",
                "Rcpp, 10 epochs"      = "BackPropNN, 10 epochs",
                "Rcpp, 100 epochs"     = "BackPropNN, 100 epochs",
                "Rcpp, 500 epochs"     = "BackPropNN, 500 epochs",
                "Rcpp, 2000 epochs"     = "BackPropNN, 2000 epochs",
                "nnet, 10 iterations"  = "nnet, 10 iterations",
                "nnet, 100 iterations" = "nnet, 100 iterations",
                "nnet, 500 iterations" = "nnet, 500 iterations",
                "nnet, 2000 iterations" = "nnet, 2000 iterations")

idx <- match(acc_lookup[names(timings)], accuracy$engine)
cost_accuracy <- data.frame(
  engine    = names(timings),
  train_sec = as.numeric(timings),
  auc       = accuracy$auc[idx],
  mse       = accuracy$mse[idx]
)

## ---- Save -----------------------------------------------------------------
results <- list(
  train         = res_train,
  train_rel     = summary(res_train, relative = TRUE),
  pred          = res_pred,
  pred_rel      = summary(res_pred, relative = TRUE),
  accuracy      = accuracy,
  cost_accuracy = cost_accuracy,
  engines_agree = engines_agree,
  settings      = list(num_obs = num_obs, i = i, h = h, o = o,
                       learning_rate = learning_rate,
                       activation_func = activation_func, E = E, n_rep = n_rep),
  meta          = list(
    date     = Sys.time(),
    r        = R.version.string,
    platform = R.version$platform,
    os       = unname(Sys.info()[["sysname"]]),
    versions = c(BackPropNN     = as.character(utils::packageVersion("BackPropNN")),
                 nnet           = as.character(utils::packageVersion("nnet")),
                 Rcpp           = as.character(utils::packageVersion("Rcpp")),
                 RcppArmadillo  = as.character(utils::packageVersion("RcppArmadillo")))
  )
)

dir.create("inst/benchmarks", recursive = TRUE, showWarnings = FALSE)
saveRDS(results, "inst/benchmarks/results.rds")

message("Saved to inst/benchmarks/results.rds")
print(res_train)
print(results$train_rel)
print(res_pred)
print(results$pred_rel)
print(accuracy, row.names = FALSE, digits = 4)
print(cost_accuracy, row.names = FALSE, digits = 4)
