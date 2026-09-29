#!/usr/bin/env Rscript
#
# Generate cross-implementation log-logistic SPEI reference fixtures with the R
# `SPEI` package (Vicente-Serrano et al., 2010; Beguería et al., 2014), the
# package the CSIC SPEIbase is built from. Invoked by
# scripts/prepare_spei_loglogistic_fixtures.py, which supplies the shared
# water-balance series and converts the CSV output to the committed .npy
# fixtures. This script is not run in CI.
#
# Usage:
#   Rscript scripts/spei_loglogistic_reference.R <input.csv> <output_dir> <scale> [<scale> ...]
#
# input.csv columns: year, month, then one column per series holding the
# (P - PET) water balance with the +1000 mm offset this library applies before
# scaling, so the two implementations fit the identical scaled series.
#
# Writes, per scale:
#   fitted_<NN>.csv  year, month, then one standardized-SPEI column per series
#   params_<NN>.csv  series, month, loc, scale, shape (the fitted GLO params)
# and once: r_versions.csv  name, version

suppressMessages({
  library(SPEI)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("usage: spei_loglogistic_reference.R <input.csv> <output_dir> <scale> [<scale> ...]", call. = FALSE)
}
input_path <- args[[1]]
output_dir <- args[[2]]
scales <- as.integer(args[-(1:2)])

d <- read.csv(input_path, check.names = FALSE)
if (!all(c("year", "month") %in% names(d))) {
  stop("input must have year and month columns", call. = FALSE)
}
if (nrow(d) %% 12 != 0) {
  stop("input must contain complete calendar years", call. = FALSE)
}
if (!all(sort(unique(d$month)) == 1:12)) {
  stop("input must cover every calendar month 1..12", call. = FALSE)
}
series_names <- setdiff(names(d), c("year", "month"))
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

for (scale in scales) {
  fitted <- matrix(NA_real_, nrow(d), length(series_names), dimnames = list(NULL, series_names))
  parameter_rows <- list()
  for (series_name in series_names) {
    values <- ts(d[[series_name]], start = c(d$year[[1]], d$month[[1]]), frequency = 12)
    fit <- spei(values, scale = scale, distribution = "log-Logistic", fit = "ub-pwm", verbose = FALSE)
    fitted[, series_name] <- as.numeric(fit$fitted)
    for (month in 1:12) {
      # coefficient rows are loc, scale, shape; a calendar step whose sample
      # cannot be fitted is reported as NA in all three, matching the NaNs this
      # library's transform emits for a zeroed scale.
      coefficients <- as.numeric(fit$coefficients[, 1, month])
      parameter_rows[[length(parameter_rows) + 1]] <- data.frame(
        series = series_name,
        month = month,
        loc = coefficients[1],
        scale = coefficients[2],
        shape = coefficients[3],
        stringsAsFactors = FALSE
      )
    }
  }
  fitted_out <- data.frame(year = d$year, month = d$month, fitted, check.names = FALSE)
  write.csv(fitted_out, file.path(output_dir, sprintf("fitted_%02d.csv", scale)), row.names = FALSE, quote = FALSE)
  write.csv(
    do.call(rbind, parameter_rows),
    file.path(output_dir, sprintf("params_%02d.csv", scale)),
    row.names = FALSE,
    quote = FALSE
  )
}

versions <- data.frame(
  name = c("R", "SPEI", "TLMoments", "lmom"),
  version = c(
    as.character(getRversion()),
    as.character(packageVersion("SPEI")),
    as.character(packageVersion("TLMoments")),
    as.character(packageVersion("lmom"))
  )
)
write.csv(versions, file.path(output_dir, "r_versions.csv"), row.names = FALSE, quote = FALSE)
