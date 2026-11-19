% REFIMPL-30 driver for JonKing93/pdsi under GNU Octave. This file is ours; upstream functions
% (pdsi, soilMoisture, zPDSI) are called unmodified from /work/upstream.
%
% "supplied-pe": the steps of upstream pdsi.m in order, with PE taken from the fixture instead of
%   computed by pethorn (no latitude is stored with the division fixtures). All 344 divisions.
% "published": upstream pdsi() end to end with its own Thornthwaite PE, division 0101 only, at the
%   latitude implied by NCEI soilcnst.asc (-tan(lat) = -0.6899). Used for the "runs" test only.
addpath('/work/upstream');
Pin  = load('/work/data/precip_in.txt');   % 344 x 1536, inches
PEin = load('/work/data/pet_in.txt');      % 344 x 1536, inches
awc  = load('/work/data/awc.txt');         % 344 x 1, inches
nDiv = size(Pin, 1); nYears = 128; nTime = 12 * nYears;
years = [1895 2022]; calibYears = [1931 1990];
yrs = years(1):years(2); calib = yrs >= calibYears(1) & yrs <= calibYears(2); nCalib = sum(calib);

X = nan(nDiv, nTime); Xm = nan(nDiv, nTime); Zall = nan(nDiv, nTime);
for k = 1:nDiv
  awcs = 1.0; awcu = awc(k) - 1.0;
  P = reshape(Pin(k, :)', [12, nYears]); PE = reshape(PEin(k, :)', [12, nYears]);
  Pcalib = P(:, calib); PEcalib = PE(:, calib);
  Pmean = mean(Pcalib, 2); PEmean = mean(PEcalib, 2);
  s = soilMoisture(repmat(Pmean, [10, 1]), repmat(PEmean, [10, 1]), awcs, awcu, awcs, awcu, 1);
  ssi = s.ssi(end-11); sui = s.sui(end-11);
  s = soilMoisture(Pcalib(:), PEcalib(:), awcs, awcu, ssi, sui, 2);
  m = @(v) mean(reshape(v, [12, nCalib]), 2);
  et = m(s.et); r = m(s.r); pr = m(s.pr); ro = m(s.ro); pro = m(s.pro); loss = m(s.loss); ploss = m(s.ploss);
  alpha = et ./ PEmean; alpha(PEmean == 0 & et == 0) = 1; alpha(PEmean == 0 & et ~= 0) = 0;
  beta = r ./ pr; beta(pr == 0 & r == 0) = 1; beta(pr == 0 & r ~= 0) = 0;
  gamma = ro ./ pro; gamma(pro == 0 & ro == 0) = 1; gamma(pro == 0 & ro ~= 0) = 0;
  delta = loss ./ ploss; delta(ploss == 0) = 0;
  s = soilMoisture(P(:), PE(:), awcs, awcu, ssi, sui, 3, false);
  prf = reshape(s.pr, [12, nYears]); prof = reshape(s.pro, [12, nYears]); plossf = reshape(s.ploss, [12, nYears]);
  Pcafec = alpha .* PE + beta .* prf + gamma .* prof - delta .* plossf;
  D = P - Pcafec; Dmean = mean(abs(D(:, calib)), 2);
  w = (PEmean + r + ro) ./ (Pmean + loss) + 2.8;
  Kprime = 1.5 * log10(w ./ Dmean) + 0.5;
  K = (17.67 / sum(Dmean .* Kprime)) .* Kprime;
  Z = K .* D;
  [x, xm] = zPDSI(Z(:), false);
  X(k, :) = x'; Xm(k, :) = xm'; Zall(k, :) = Z(:)';
  if mod(k, 50) == 0, printf('%d/%d\n', k, nDiv); end
end
save('-ascii', '/work/out/supplied_pe_X.txt', 'X');
save('-ascii', '/work/out/supplied_pe_Xm.txt', 'Xm');
save('-ascii', '/work/out/supplied_pe_Z.txt', 'Zall');
printf('supplied-pe finite X: %d of %d\n', sum(isfinite(X(:))), numel(X));

T = load('/work/data/temp_c_0101.txt');
lat = atand(0.6899);
[Xp, Xmp, Zp] = pdsi(T, Pin(1, :)' * 25.4, years, lat, 25.4, (awc(1) - 1) * 25.4, calibYears);
save('-ascii', '/work/out/published_0101_X.txt', 'Xp');
save('-ascii', '/work/out/published_0101_Xm.txt', 'Xmp');
save('-ascii', '/work/out/published_0101_Z.txt', 'Zp');
printf('published (0101) finite X: %d of %d\n', sum(isfinite(Xp(:))), numel(Xp));
