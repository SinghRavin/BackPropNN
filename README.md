
<!-- README.md is generated from README.Rmd. Please edit that file -->

# BackPropNN

<!-- badges: start -->

[![R-CMD-check](https://github.com/SinghRavin/BackPropNN/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/SinghRavin/BackPropNN/actions/workflows/R-CMD-check.yaml)

<!-- badges: end -->

The neural network algorithm trained by the process of backpropagation
is implemented. The stochastic gradient descent (SGD) is adopted for
this algorithm i.e., the weight and bias matrices are updated after
learning from the errors of each data set point one by one. In this
package, 3 layer (input, one hidden and output) neural network is
considered. The mathematical equation involved are given below,

**Feed Forward:**

\[H\] = sigma(\[W_IH\].\[I\] + \[B_H\])

\[O\] = sigma(\[W_HO\].\[H\] + \[B_O\])

**Backpropagation:**

\[delta_W_HO\] =
(learning_rate)\[Output_Errors\]x\[O(1-O)\].\[H_tranpose\]

\[delta_W_IH\] =
(learning_rate)\[Hiddden_Errors\]x\[H(1-H)\].\[I_tranpose\]

\[delta_B_O\] = (learning_rate)\[Output_Errors\]x\[O(1-O)\]

\[delta_B_H\] = (learning_rate)\[Hidden_Errors\]x\[H(1-H)\]

Where, \[\] represents the matrix, x represents the Hadamard
multiplication (element-wise), and . represents the usual matrix
multiplication. H represents hidden matrix, O represents output matrix,
W_IH represents weight matrix between input layer and hidden layer, W_HO
represents weight matrix between hidden layer and output layer, B_H
represents bias matrix for hidden layer, B_O represents bias matrix for
output layer. The chosen activation function are Sigmoid or ReLU.

Weights are initialised randomly with variance 1/fan-in, and biases at
zero. Both the R and the C++ engines draw from R’s RNG stream, so
`set.seed()` gives identical weights in either engine. The `epochs`
argument controls how many complete passes are made over the data; it
defaults to 1.

## Installation

You can install the development version of BackPropNN from GitHub like
so:

``` r
# install.packages("devtools")
devtools::install_github("SinghRavin/BackPropNN")
```

## Example

This is a basic example which shows you how to solve a common problem:

# Simulated data - Logistics data.

``` r
library(BackPropNN) # Loading the package.

set.seed(100)
num_obs <- 10000 # Number of observations.

# Setting coefficients values for the logit function.
beta0 <- -13.6
beta1 <- 0.10
beta2 <- 0.05

# Simulating the independent variables.
X1 <- runif(n=num_obs, min=18, max=60)
X2 <- runif(n=num_obs, min=100, max=250)
prob <- exp(beta0 + beta1*X1 + beta2*X2) / (1 + exp(beta0 + beta1*X1 + beta2*X2))

# Generating binary outcome variable.
Yvec <- rbinom(n=num_obs, size=1, prob=prob)

# Predictors are standardised. A sigmoid saturates when inputs are on the raw
# scale (X2 averages 175), which stalls learning in every engine compared here.
data <- data.frame(X1 = as.numeric(scale(X1)),
                   X2 = as.numeric(scale(X2)),
                   Y  = Yvec)
```

# Running the functions of BackPropNN package.

``` r
i <- 2 # number of input nodes
h <- 4 # number of hidden nodes
o <- 1 # number of output nodes
learning_rate <- 0.1 # The learning rate of the algorithm
activation_func <- "sigmoid" # the activation function
epochs <- 10 # number of complete passes over the data

set.seed(1)
nn_model <- back_propagation_training(i, h, o, learning_rate, activation_func,
                                      data, epochs = epochs)
```

# Summarizing the results of nn_model.

``` r
nn_model
#> A 3-layer neural network trained by backpropagation
#> 
#>   Input nodes   : 2 
#>   Hidden nodes  : 4 
#>   Output nodes  : 1 
#>   Activation    : sigmoid 
#>   Learning rate : 0.1 
#>   Training rows : 10000 
#>   In-sample MSE : 0.1204
summary(nn_model)
#> $num_nodes
#>  # of input nodes # of hidden nodes # of output nodes 
#>                 2                 4                 1 
#> 
#> $activation_function
#> [1] "sigmoid"
#> 
#> $learning_rate
#> [1] 0.1
#> 
#> $weight_bias_matrices
#> $weight_bias_matrices$weight_input_hidden
#>             X1        X2
#> [1,]  1.119254  2.149039
#> [2,] -2.313731 -3.688991
#> [3,]  1.239835  2.840185
#> [4,]  2.170264  2.719256
#> 
#> $weight_bias_matrices$weight_hidden_output
#>           [,1]      [,2]     [,3]     [,4]
#> [1,] 0.9573678 -2.354976 1.608131 1.676876
#> 
#> $weight_bias_matrices$bias_hidden
#>           [,1]
#> [1,] -1.440733
#> [2,] -2.186461
#> [3,] -3.058938
#> [4,] -2.271660
#> 
#> $weight_bias_matrices$bias_output
#>           [,1]
#> [1,] -1.066418
```

## How the comparison is set up

`nnet` optimises with BFGS: one weight update per iteration, each using
a gradient over the whole data set. `BackPropNN` uses stochastic
gradient descent: one update per observation. An epoch and an iteration
are therefore not the same unit of work, and the two are compared under
two protocols:

- **Speed** at an equal number of passes over the data. A pass costs the
  same arithmetic in every engine, so this isolates implementation
  efficiency.
- **Accuracy** with each engine run to convergence under its own
  optimiser.

Every number below comes from the same 70/30 split (7,000 train, 3,000
test), the same seeds, and the same high-resolution clock. Produced by
[`inst/benchmarks/fair_benchmark.R`](inst/benchmarks/fair_benchmark.R)
in one run on 20 July 2026 (Windows, R version 4.4.2 (2024-10-31 ucrt),
nnet 7.3.19), and read from a saved file so the published figures do not
drift between builds.

# Speed: training, at an equal number of passes

``` r
results$train
#> # A data frame: 3 × 6
#>   expression      min   median `itr/sec` mem_alloc `gc/sec`
#>   <bch:expr> <bch:tm> <bch:tm>     <dbl> <bch:byt>    <dbl>
#> 1 Pure R        5.13s     5.9s     0.154  196.47KB     6.12
#> 2 Rcpp         40.2ms   45.8ms    20.6    331.62KB     0   
#> 3 nnet        94.78ms  111.9ms     8.51     1.04MB     0
results$train_rel
#> # A data frame: 3 × 6
#>   expression    min median `itr/sec` mem_alloc `gc/sec`
#>   <bch:expr>  <dbl>  <dbl>     <dbl>     <dbl>    <dbl>
#> 1 Pure R     128.   129.         1        1         Inf
#> 2 Rcpp         1      1        134.       1.69      NaN
#> 3 nnet         2.36   2.44      55.3      5.45      NaN
```

# Speed: prediction

``` r
results$pred
#> # A data frame: 3 × 6
#>   expression      min   median `itr/sec` mem_alloc `gc/sec`
#>   <bch:expr> <bch:tm> <bch:tm>     <dbl> <bch:byt>    <dbl>
#> 1 Pure R     257.41ms 296.65ms      3.30    98.6KB     8.67
#> 2 Rcpp        983.2µs 998.55µs    992.      25.9KB     0   
#> 3 nnet         1.02ms   1.06ms    897.       338KB     0
results$pred_rel
#> # A data frame: 3 × 6
#>   expression    min median `itr/sec` mem_alloc `gc/sec`
#>   <bch:expr>  <dbl>  <dbl>     <dbl>     <dbl>    <dbl>
#> 1 Pure R     262.   297.          1       3.80      Inf
#> 2 Rcpp         1      1         301.      1         NaN
#> 3 nnet         1.04   1.06      272.     13.0       NaN
```

The R and C++ engines are the same algorithm, so given the same seed
they must produce the same weights. The benchmark script asserts this:
TRUE.

# Accuracy

Held-out AUC and MSE, averaged over 5 random initialisations. Both
engines are run well past the point of improvement, so convergence is
demonstrated rather than assumed: `BackPropNN` is flat from 10 epochs
onward, and `nnet` is identical at 500 and 2000 iterations because BFGS
reaches its tolerance and stops early.

``` r
knitr::kable(results$accuracy, digits = 4)
```

| engine                  |    auc |    mse |
|:------------------------|-------:|-------:|
| BackPropNN, 10 epochs   | 0.9004 | 0.1253 |
| BackPropNN, 100 epochs  | 0.9010 | 0.1251 |
| BackPropNN, 500 epochs  | 0.9008 | 0.1250 |
| BackPropNN, 2000 epochs | 0.9006 | 0.1252 |
| nnet, 10 iterations     | 0.8558 | 0.1613 |
| nnet, 100 iterations    | 0.8865 | 0.1297 |
| nnet, 500 iterations    | 0.8933 | 0.1268 |
| nnet, 2000 iterations   | 0.8933 | 0.1268 |

# ROC curves on held-out data

Each engine at convergence: 100 epochs for `BackPropNN`, 500 BFGS
iterations for `nnet`. Single seed, so these AUCs sit close to but not
exactly on the averaged figures above.

``` r
set.seed(100)
train_idx <- sample(seq_len(num_obs), size = round(0.7 * num_obs))
train <- data[train_idx, ]
test  <- data[-train_idx, ]

X_tr <- as.matrix(train[, 1:2]); Y_tr <- as.matrix(train[, 3])
X_te <- as.matrix(test[, 1:2])

set.seed(1)
fit_bp <- back_propagation_training_rcpp(i, h, o, learning_rate, activation_func,
                                         as.matrix(train), epochs = 100)
set.seed(1)
fit_nnet <- nnet::nnet(X_tr, Y_tr, size = h, maxit = 500, trace = FALSE)

invisible(pROC::roc(test$Y, as.numeric(feed_forward_rcpp(test, fit_bp)),
                    plot = TRUE, print.auc = TRUE, quiet = TRUE,
                    main = "ROC curve by BackPropNN (held-out)"))
```

<img src="man/figures/README-unnamed-chunk-8-1.png" width="100%" />

``` r

invisible(pROC::roc(test$Y, as.numeric(stats::predict(fit_nnet, X_te, type = "raw")),
                    plot = TRUE, print.auc = TRUE, quiet = TRUE,
                    main = "ROC curve by R nnet (held-out)"))
```

<img src="man/figures/README-unnamed-chunk-8-2.png" width="100%" />

# Cost against accuracy

What accuracy each engine reaches, and what it cost. Directly comparable
regardless of the SGD/BFGS difference. Training time is the median of
five seeded fits.

``` r
knitr::kable(results$cost_accuracy, digits = 4, row.names = FALSE)
```

| engine                | train_sec |    auc |    mse |
|:----------------------|----------:|-------:|-------:|
| Pure R, 10 epochs     |    7.1554 | 0.9004 | 0.1253 |
| Rcpp, 10 epochs       |    0.0473 | 0.9004 | 0.1253 |
| Rcpp, 100 epochs      |    0.4375 | 0.9010 | 0.1251 |
| Rcpp, 500 epochs      |    2.3451 | 0.9008 | 0.1250 |
| Rcpp, 2000 epochs     |    9.5348 | 0.9006 | 0.1252 |
| nnet, 10 iterations   |    0.0771 | 0.8558 | 0.1613 |
| nnet, 100 iterations  |    0.5607 | 0.8865 | 0.1297 |
| nnet, 500 iterations  |    1.3427 | 0.8933 | 0.1268 |
| nnet, 2000 iterations |    1.3400 | 0.8933 | 0.1268 |

# What these results show

**The C++ rewrite removed the interpreter, not the arithmetic.** At an
equal number of passes the Rcpp engine is about 129 times faster than
the pure R engine, and about 2.4 times faster than `nnet`. Prediction is
a tie with `nnet`. The pure R engine allocates an S3 list on every
activation call, resolves a closure on every call, and coerces matrices
once per row; its garbage collector runs hundreds of times per
benchmark, while the C++ engine triggers it zero times and allocates
less memory than `nnet`. The algorithm is unchanged between the two
engines: same initialisation, same update order, same equations.

**Both engines converge, to different optima.** `BackPropNN` settles at
an AUC near 0.901 and does not move between 10 and 2000 epochs. `nnet`
settles near 0.893, reached by 500 iterations and unchanged at 2000. The
from-scratch implementation converges to the better optimum, and reaches
it roughly 28 times faster.

**Most of that 28x is the optimiser, not the language.** Ten passes is
100,000 weight updates for SGD and ten for BFGS, so `BackPropNN` needs
50 times fewer passes to converge. Multiply that by the 2.4x per-pass
advantage and the gap follows. This holds for a 17-parameter network on
well-separated, linearly generated data, which is where SGD is
strongest. On a harder surface BFGS’s curvature information is exactly
what pays off, and the ranking could reverse.

**Timings vary with system load.** They come from a single Windows
machine, so the ordering is reliable and the exact multiples are
approximate.

# Known limitations

- The output layer uses the same activation as the hidden layer, so ReLU
  is not appropriate for binary classification with this implementation.
- The number of output nodes is nominally an argument but the prediction
  path assumes a single output.
- There is no convergence check, learning-rate schedule, or
  mini-batching; the training loop runs for exactly the number of epochs
  requested.
- Predictors are not standardised internally. As shown above, this
  matters a great deal for sigmoid activations, and is left to the user.
