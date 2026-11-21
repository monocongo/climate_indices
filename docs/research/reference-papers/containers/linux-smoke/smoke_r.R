# R comparator smoke calls. Prints `label<TAB>June-2005 value` lines (NA stays NA), at full
# precision so run.sh can round to the decimals expected.tsv records.
# Inputs: /work/inputs.csv (see inputs.py). Calls and arguments follow the "Smoke evidence"
# table in comparators.md.
suppressMessages({ library(SPEI); library(SCI); library(SEI); library(climatekit); library(Evapotranspiration) })

d <- read.csv("/work/inputs.csv", na.strings = "NA")
n <- nrow(d)
jun05 <- 1325 + 1  # 1-based index of 2005-06
p <- ts(d$precip, start = c(1895, 1), frequency = 12)
tave <- ts(d$tmean, start = c(1895, 1), frequency = 12)
pet_r <- thornthwaite(tave, 25.2292, na.rm = TRUE, verbose = FALSE)  # PET for the SPEI call: R SPEI's own Thornthwaite
wb <- ts(d$precip - as.numeric(pet_r), start = c(1895, 1), frequency = 12)
cal <- c(1981, 1); calend <- c(2010, 12)
lat <- 25.2292

out <- function(label, x) cat(label, "\t", if (is.na(x)) "NA" else sprintf("%.17g", x), "\n", sep = "")
at <- function(x) as.numeric(x)[jun05]
try_out <- function(label, expr) tryCatch(out(label, expr), error = function(e) { message(label, ": ", conditionMessage(e)); cat(label, "\tERROR\n", sep = "") })

try_out("R_SPEI_spi3_gamma_ubpwm", at(spi(p, 3, distribution = "Gamma", fit = "ub-pwm", ref.start = cal, ref.end = calend, na.rm = TRUE, verbose = FALSE)$fitted))
try_out("R_SPEI_spi3_gamma_maxlik", at(spi(p, 3, distribution = "Gamma", fit = "max-lik", ref.start = cal, ref.end = calend, na.rm = TRUE, verbose = FALSE)$fitted))
try_out("R_SPEI_spei3_loglogistic_ubpwm", at(spei(wb, 3, distribution = "log-Logistic", fit = "ub-pwm", ref.start = cal, ref.end = calend, na.rm = TRUE, verbose = FALSE)$fitted))
try_out("R_SPEI_thornthwaite", at(pet_r))

# SCI: fit on 1981-2010 (3-month sums), transform the full series.
try_out("R_SCI_gamma", {
  i <- which(time(p) >= 1981 & time(p) < 2011)
  fit <- fitSCI(as.numeric(p)[i], first.mon = 1, time.scale = 3, distr = "gamma", p0 = TRUE)
  at(transformSCI(as.numeric(p), first.mon = 1, obj = fit))
})

# SEI: gamma MLE on 3-month sums, fitted to June values only (one distribution per calendar
# month), reference 1981-2010. Fitting all months with one distribution gives 0.508169, not
# 0.343431; the macOS call was not saved, so this is the reading that reproduces its value.
try_out("R_SEI_gamma_mle", {
  s3 <- as.numeric(stats::filter(d$precip, rep(1, 3), sides = 1))
  yr <- as.integer(substr(d$date, 1, 4)); mon <- as.integer(substr(d$date, 6, 7))
  ref <- yr >= 1981 & yr <= 2010
  z <- suppressWarnings(std_index(x_new = s3[mon == 6], x_ref = s3[ref & mon == 6], dist = "gamma", method = "mle", lower = 0, cens = "none"))
  z[sum(mon[1:jun05] == 6)]
})

try_out("R_climatekit_spi3_gamma", { k <- ck_spi(as.numeric(p), as.Date(d$date), scale = 3, distribution = "gamma"); k$value[k$period == as.Date("2005-06-01")] })

# Evapotranspiration: the package's own bundled data, first month.
try_out("R_Evapotranspiration_HargreavesSamani", {
  data("processeddata"); data("constants")
  r <- ET.HargreavesSamani(data = processeddata, constants = constants, ts = "monthly", message = "no", save.csv = "no")
  as.numeric(r$ET.Monthly)[1]
})
