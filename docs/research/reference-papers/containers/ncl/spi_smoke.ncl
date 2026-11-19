; One-value smoke test for dim_spi_n (REFIMPL-03). Reads /work/p.txt: 1476 monthly
; precipitation totals, January 1895 to December 2017 (mm), -999 for missing.
; NCL has no calibration-period argument: the fit uses the whole record.
begin
  n   = 1476
  p   = asciiread("/work/p.txt", (/n/), "double")
  p@_FillValue = -999.d0
  t   = (2005 - 1895) * 12 + 5          ; June 2005, zero-based index 1325
  opt = False
  spi_gamma = dim_spi_n(p, 3, opt, 0)
  opt = True
  opt@spi_type = 3                       ; Pearson type III
  spi_p3 = dim_spi_n(p, 3, opt, 0)
  print("NCL dim_spi_n SPI-3 gamma   June 2005 = " + sprintf("%9.6f", spi_gamma(t)))
  print("NCL dim_spi_n SPI-3 Pearson June 2005 = " + sprintf("%9.6f", spi_p3(t)))
end
