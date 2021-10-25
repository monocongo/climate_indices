from dataclasses import fields

import pytest

from climate_indices.fittings import GammaFit, PearsonFit
FITCLASSES = [GammaFit, PearsonFit]

@pytest.mark.parametrize("fitclass", FITCLASSES)
@pytest.mark.parametrize("paramdict", [
        {"alpha": 0.4, "beta": 4.2, "skew": 0.74, "loc": 12.0,
            "scale": 9.1, "prob_zero": 0.0},
        {"alphas": 0.56, "betas": 3.12, "skews": 0.31, "locs": 10.2,
            "scales": 2.3, "probabilities_of_zero": 0.4}
])
def test_from_dict(fitclass, paramdict):
    params = fitclass.from_dict(paramdict)
    normmap = dict(fitclass._altnames)
    for f in fields(fitclass):
        assert getattr(params, f.name) == paramdict.get(
                f.name, paramdict.get(normmap[f.name], None))

@pytest.mark.parametrize("fitclass", FITCLASSES)
def test_from_none(fitclass):
    params = fitclass.from_dict(None)
    for f in fields(fitclass):
        assert getattr(params, f.name) is None

@pytest.mark.parametrize("fitclass", FITCLASSES)
def test_bad_dict(fitclass):
    really_bad_dict = {'waka': 4.3, 'foo': -384.2, 'beta': 0.3, 'loc': 19.2}
    with pytest.raises(KeyError):
        fitclass.from_dict(really_bad_dict)


