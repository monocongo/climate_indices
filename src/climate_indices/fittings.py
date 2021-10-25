from dataclasses import dataclass

from climate_indices import compute


class FittingParams:
    @classmethod
    def from_dict(klass, params):
        """
        Compatibility shim. Convert old accepted parameter dictionaries
        and return specific instances of FittingParams subclasses.
        Don't call this method from FittingParams, call it from the subclass
        you want.
        
        See https://github.com/monocongo/climate_indices/issues/449
        """
        if klass is FittingParams:
            raise NotImplementedError("This is a base class. Call .from_dict()"
                                      " from a class like PearsonFit.")

        if params is None:
            return klass()

        normed = {}
        for name, altname in klass._altnames:
            val = params.get(name, None)
            if val is None:
                if altname not in params:
                    raise KeyError(f"Missing key {name}, or its alternate, {altname}")
                val = params[altname]
            normed[name] = val
        return klass(**normed)

    @classmethod
    def fit(klass, data):
        """
        Given the 1-D numpy array, return an instance of a FittingParams subclass
        with the estimated distribution parameters.

        Don't call this method from FittingParams, call it from the subclass
        you want.
        """
        raise NotImplementedError(f"{klass} hasn't implemented fit() yet...")

    def transform(self, values, data_start_year,
            calibration_start_year, calibration_end_year,
            periodicity):
        """
        Normalize data values according to the `FittingParams`'s instance parameters.
        If no parameters have been set previously, then parameters will be estimated
        but not populated in this data object (yet).
        """
        raise NotImplementedError(f"{klass} hasn't implemented transform() yet...")
                

@dataclass
class GammaFit(FittingParams):
    alpha: float = None
    beta: float = None

    _altnames = (("alpha", "alphas"), ("beta", "betas"))

    def transform(self, values, data_start_year,
            calibration_start_year, calibration_end_year,
            periodicity):
        """
        Normalize data values according to the Gamma Distribution parameters.
        If no parameters have been set previously, then parameters will be estimated
        but not populated in this data object (yet).
        """
        # fit the scaled values to a gamma distribution
        # and transform to corresponding normalized sigmas
        values = compute.transform_fitted_gamma(
            values,
            data_start_year,
            calibration_start_year,
            calibration_end_year,
            periodicity,
            self.alpha,
            self.beta,
        )

        return values

@dataclass
class PearsonFit(FittingParams):
    skew: float = None
    loc: float = None
    scale: float = None
    prob_zero: float = None

    _altnames = (("skew", "skews"), ("scale", "scales"), ("loc", "locs"),
                 ("prob_zero", "probabilities_of_zero"))

    def transform(self, values, data_start_year,
            calibration_start_year, calibration_end_year,
            periodicity):
        """
        Normalize data values according to the Pearson III Distribution parameters.
        If no parameters have been set previously, then parameters will be estimated
        but not populated in this data object (yet).
        """
        # fit the scaled values to a Pearson Type III distribution
        # and transform to corresponding normalized sigmas
        values = compute.transform_fitted_pearson(
            values,
            data_start_year,
            calibration_start_year,
            calibration_end_year,
            periodicity,
            self.prob_zero,
            self.loc,
            self.scale,
            self.skew
        )

        return values

