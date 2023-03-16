# context("Basic set of tests") # This will generate a warning as the current
# version is deprecated
test_that("map(func,a) = func(a)", {
  # Preparing the test
  a <- matrix(1:9,3,3)
  func <- function(x){
    return(x+2)
    }

  # Calling the function
  ans0 <- matrix(3:11,3,3)
  ans1 <- map(func,a)

  # Are these equal?
  expect_equal(ans0, ans1$mapped_matrix)
})
