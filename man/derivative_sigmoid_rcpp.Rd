% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{derivative_sigmoid_rcpp}
\alias{derivative_sigmoid_rcpp}
\title{The title of -derivative_sigmoid_rcpp-}
\usage{
derivative_sigmoid_rcpp(a)
}
\arguments{
\item{a}{Numeric matrix.}
}
\value{
A matrix.
}
\description{
Here is a brief description
}
\details{
Computes the derivative of sigmoid function value of each element of matrix \code{a}.
}
\examples{
mat <- matrix(1:9,3,3)
derivative_sigmoid_rcpp(mat)

}
