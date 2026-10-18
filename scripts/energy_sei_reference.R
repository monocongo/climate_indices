#!/usr/bin/env Rscript
# Run the pinned SEI implementation, not a translation of its statistical code.
# Usage: energy_sei_reference.R <temporary-library> <input.csv> <output-directory>
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("Expected library, input.csv and output directory")
.libPaths(c(args[[1]], .libPaths()))
suppressPackageStartupMessages(library(SEI))
stopifnot(as.character(packageVersion("SEI")) == "0.2.0")
stopifnot(as.character(packageVersion("fitdistrplus")) == "1.2.6")
d <- read.csv(args[[2]], na.strings = c("nan", "NA"))
out <- args[[3]]
# write.csv otherwise limits numeric output to 15 significant digits.
write_reference <- function(data, path) {
  data[] <- lapply(data, function(x) formatC(x, digits = 17, format = "g"))
  write.csv(data, path, row.names = FALSE, quote = FALSE)
}
warnings_seen <- character()
record_warnings <- function(expr) {
  withCallingHandlers(expr, warning = function(w) {
    warnings_seen <<- c(warnings_seen, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
}

# The grouped SEI recursion omits some arguments; aggregate and group explicitly.
for (id in unique(d$case)) {
  x <- d[d$case == id, ]
  scale <- x$scale[[1]]
  aggregation <- x$aggregation[[1]]
  scaled <- vapply(seq_len(nrow(x)), function(i) {
    if (i < scale) return(NA_real_)
    window <- x$value[(i - scale + 1):i]
    if (anyNA(window)) return(NA_real_)
    total <- sum(window)
    if (aggregation == "mean") total / scale else total
  }, numeric(1))
  probability <- normal <- count <- rep(NA_real_, nrow(x))
  for (month in 1:12) {
    ind <- x$month == month
    calibration <- ind & x$year >= x$calibration_start & x$year <= x$calibration_end
    reference <- scaled[calibration]
    count[ind] <- sum(!is.na(reference))
    probability[ind] <- record_warnings(SEI::std_index(
      x_new = scaled[ind], x_ref = reference, dist = "empirical", method = "mle",
      index_type = "prob01", lower = -Inf, upper = Inf, cens = "none", n_thres = 10
    ))
    normal[ind] <- record_warnings(SEI::std_index(
      x_new = scaled[ind], x_ref = reference, dist = "empirical", method = "mle",
      index_type = "normal", lower = -Inf, upper = Inf, cens = "none", n_thres = 10
    ))
  }
  write_reference(data.frame(scaled, probability, normal, count),
                  file.path(out, paste0(id, ".csv")))
}

# Supported parametric control: gamma MLE on strictly positive production.
# The saved parameters isolate the CDF from the project's different gamma fit.
x <- d[d$case == "production_sum_1", ]
probability <- normal <- shape <- rate <- rep(NA_real_, nrow(x))
for (month in 1:12) {
  ind <- x$month == month
  calibration <- ind & x$year >= x$calibration_start & x$year <= x$calibration_end
  fit <- record_warnings(SEI::std_index(
    x_new = x$value[ind], x_ref = x$value[calibration], dist = "gamma", method = "mle",
    index_type = "prob01", lower = -Inf, upper = Inf, cens = "none", n_thres = 10,
    return_fit = TRUE
  ))
  probability[ind] <- fit$si
  shape[ind] <- fit$params[["shape"]]
  rate[ind] <- fit$params[["rate"]]
  normal[ind] <- record_warnings(SEI::std_index(
    x_new = x$value[ind], x_ref = x$value[calibration], dist = "gamma", method = "mle",
    index_type = "normal", lower = -Inf, upper = Inf, cens = "none", n_thres = 10
  ))
}
write_reference(data.frame(probability, normal, shape, rate), file.path(out, "gamma_control.csv"))

# With no censored calibration values, SEI leaves finite-bound evaluations NA.
reference <- seq(0.1, 0.9, length.out = 100)
evaluated <- c(0, 0.5, 1)
# std_index rejects a numeric censoring pair; call its PIT primitive directly.
probability <- record_warnings(SEI::get_pit(
  x_new = evaluated, x_ref = reference, dist = "empirical", method = "mle",
  lower = 0, upper = 1, cens = c(0.1, 0.9), n_thres = 10
))
with_endpoints <- record_warnings(SEI::get_pit(
  x_new = evaluated, x_ref = c(0, reference, 1), dist = "empirical", method = "mle",
  lower = 0, upper = 1, cens = c(0.1, 0.9), n_thres = 10
))
write_reference(data.frame(evaluated, probability, with_endpoints), file.path(out, "finite_bounds.csv"))
writeLines(capture.output(sessionInfo()), file.path(out, "sessionInfo.txt"))
writeLines(unique(warnings_seen), file.path(out, "warnings.txt"))
