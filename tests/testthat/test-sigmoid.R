# context("Basic set of tests") # This will generate a warning as the current
# version is deprecated
test_that("sigmoid(a) = 1/(1+exp(-a))", {
  # Preparing the test
  a <- 2

  # Calling the function
  ans0 <- 1/(1+exp(-a))
  ans1 <- sigmoid(a)

  # Are these equal?
  expect_equal(ans0, ans1$value)
})

# test_that("Plot returns -BackPropNN_foo-", {
#   expect_s3_class(
#     plot(sigmoid(2)), "BackPropNN_foo"
#   ) # This will generate an error as the function plot.funnypkg_addnum
#   # does not return an object of that class.
# })
