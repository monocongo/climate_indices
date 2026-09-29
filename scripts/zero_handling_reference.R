#!/usr/bin/env Rscript
#
# Generate cross-implementation zero-placement reference fixtures with the R
# `SEI` (Allen & Otero, 2024) and `SCI` (Gudmundsson & Stagge, 2014) packages.
# Invoked by scripts/prepare_zero_handling_fixtures.py, which supplies the
# shared input series and converts the CSV output to the committed .npy
# fixtures. This script is not run in CI.
#
# Usage:
#   Rscript scripts/zero_handling_reference.R <input.csv> <output_dir>
#
# input.csv columns: year, month, value (contiguous monthly rows, 1..12)

suppressMessages({
  library(SEI)
  library(SCI)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: zero_handling_reference.R <input.csv> <output_dir>", call. = FALSE)
}
input_path <- args[[1]]
output_dir <- args[[2]]
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

d <- read.csv(input_path)
if (!all(c("year", "month", "value") %in% names(d))) {
  stop("input must have year, month, and value columns", call. = FALSE)
}
if (nrow(d) %% 12 != 0) {
  stop("input must contain complete calendar years", call. = FALSE)
}
if (!all(sort(unique(d$month)) == 1:12)) {
  stop("input must cover every calendar month 1..12", call. = FALSE)
}
month_factor <- factor(d$month, levels = 1:12)
n_years <- nrow(d) / 12

# `std_index`'s grouping returns one matrix row per group (calendar month) and
# one column per year; write that layout straight through.
write_by_month <- function(mat, path) {
  out <- as.data.frame(t(mat))
  names(out) <- paste0("M", 1:12)
  write.csv(out, path, row.names = FALSE, quote = FALSE)
}

# SEI: in-sample gamma MLE by fitdistrplus, `lower = 0` censors precipitation
# below zero, and `cens` picks the censored-PIT constant. The three values
# "none", "prob", and "normal" are classic, centre-of-mass, and mean-zero.
# `cens` is a transform choice, so the fitted parameters must be identical
# across the three modes; stop rather than publish a mode-dependent fit.
sei_shapes <- NULL
sei_rates <- NULL
for (cens_mode in c("none", "prob", "normal")) {
  res <- std_index(
    d$value,
    gr_new = month_factor,
    gr_ref = month_factor,
    dist = "gamma",
    method = "mle",
    lower = 0,
    cens = cens_mode,
    n_thres = 10,
    return_fit = TRUE
  )
  write_by_month(res$si, file.path(output_dir, paste0("sei_", cens_mode, ".csv")))

  shapes <- vapply(res$params, function(p) if (length(p) == 2) p[["shape"]] else NA_real_, numeric(1))
  rates <- vapply(res$params, function(p) if (length(p) == 2) p[["rate"]] else NA_real_, numeric(1))
  if (is.null(sei_shapes)) {
    sei_shapes <- shapes
    sei_rates <- rates
  } else {
    stopifnot(isTRUE(all.equal(shapes, sei_shapes)), isTRUE(all.equal(rates, sei_rates)))
  }
}
write.csv(
  data.frame(month = 1:12, shape = sei_shapes, rate = sei_rates),
  file.path(output_dir, "sei_params.csv"),
  row.names = FALSE, quote = FALSE
)

# SCI: `p0.center.mass = TRUE` estimates the zero probability with the Weibull
# plotting position and places the zero mass centre accordingly. Only the
# placement constant is written; SCI's transform shares the mixed-distribution
# form but its transformSCI output is not part of the committed fixture.
sci_fit <- fitSCI(
  d$value,
  first.mon = 1,
  time.scale = 1,
  distr = "gamma",
  p0 = TRUE,
  p0.center.mass = TRUE,
  warn = FALSE
)
write.csv(
  data.frame(month = 1:12, p0_center_mass = as.numeric(sci_fit$dist.para["P0", ])),
  file.path(output_dir, "sci_p0_center_mass.csv"),
  row.names = FALSE, quote = FALSE
)

# versions, for the provenance record
write.csv(
  data.frame(
    name = c("R", "SEI", "SCI"),
    version = c(
      paste(R.version$major, R.version$minor, sep = "."),
      as.character(packageVersion("SEI")),
      as.character(packageVersion("SCI"))
    )
  ),
  file.path(output_dir, "r_versions.csv"),
  row.names = FALSE, quote = FALSE
)
