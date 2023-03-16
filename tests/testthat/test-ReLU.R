# context("Basic set of tests") # This will generate a warning as the current
# version is deprecated
test_that("ReLU(a) = max(0,a)", {
  # Preparing the test
  a <- 2.5

  # Calling the function
  ans0 <- max(0,a)
  ans1 <- ReLU(a)

  # Are these equal?
  expect_equal(ans0, ans1$value)
})

