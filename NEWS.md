# BackPropNN 0.2-0

## Breaking changes

* Weights are now initialised randomly with variance 1/fan-in rather than set
  to the constant 0.01. Under the old scheme every hidden unit computed the
  same value and received the same gradient, so a network with `h` hidden nodes
  behaved as though it had one. Results from earlier versions will not
  reproduce. Both engines draw from R's RNG stream, so `set.seed()` gives
  identical weights in the R and C++ versions.
* Biases now initialise to zero rather than 0.01.

## New features

* `back_propagation_training()` and `back_propagation_training_rcpp()` gain an
  `epochs` argument controlling the number of complete passes over the data.
  Defaults to 1, matching previous behaviour.
* Added `inst/benchmarks/fair_benchmark.R`, which runs every comparison
  reported in the README and saves the results to `inst/benchmarks/results.rds`.

## Documentation

* README rewritten. Speed is now compared at an equal number of passes over the
  data, and accuracy with each engine run to convergence under its own
  optimiser. Earlier versions compared a single SGD pass against `nnet`'s
  default 100 BFGS iterations, which understated `nnet`'s work by two orders of
  magnitude.
* README figures are read from a saved benchmark run rather than regenerated at
  knit time, so published numbers are stable between builds.
* Vignette now covers usage only; predictors are standardised in the examples,
  since sigmoid activations saturate on raw-scale inputs.
* Added a known limitations section to the README.

# BackPropNN 0.1-0

* Initial version.
