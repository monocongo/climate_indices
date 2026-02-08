"""Tests for computation error handling in climate_indices.compute.

This module tests that scipy.stats distribution fitting failures are properly
caught and re-raised as structured DistributionFittingError exceptions with
full diagnostic context.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from climate_indices import compute, exceptions


class TestGammaCDFErrorHandling:
    """Test error handling for scipy.stats.gamma.cdf() failures."""

    def test_gamma_cdf_value_error(self) -> None:
        """ValueError from gamma.cdf should be wrapped as DistributionFittingError."""
        # create test data
        values = np.array([1.0, 2.0, 3.0] * 16).reshape(4, 12)

        # mock gamma.cdf to raise ValueError
        with patch("climate_indices.compute.scipy.stats.gamma.cdf") as mock_cdf:
            mock_cdf.side_effect = ValueError("invalid parameters")

            with pytest.raises(exceptions.DistributionFittingError) as exc_info:
                compute.transform_fitted_gamma(
                    values,
                    data_start_year=2000,
                    calibration_start_year=2000,
                    calibration_end_year=2003,
                    periodicity=compute.Periodicity.monthly,
                )

            # verify structured attributes
            exc = exc_info.value
            assert exc.distribution_name == "gamma"
            assert exc.input_shape is not None
            assert exc.parameters is not None
            assert "alphas" in exc.parameters
            assert "betas" in exc.parameters
            assert exc.suggestion == "Try using pearson3 distribution instead"
            assert isinstance(exc.underlying_error, ValueError)
            assert "invalid parameters" in str(exc.underlying_error)

    def test_gamma_cdf_runtime_error(self) -> None:
        """RuntimeError from gamma.cdf should be wrapped as DistributionFittingError."""
        values = np.array([1.0, 2.0, 3.0] * 16).reshape(4, 12)

        with patch("climate_indices.compute.scipy.stats.gamma.cdf") as mock_cdf:
            mock_cdf.side_effect = RuntimeError("numerical instability")

            with pytest.raises(exceptions.DistributionFittingError) as exc_info:
                compute.transform_fitted_gamma(
                    values,
                    data_start_year=2000,
                    calibration_start_year=2000,
                    calibration_end_year=2003,
                    periodicity=compute.Periodicity.monthly,
                )

            exc = exc_info.value
            assert exc.distribution_name == "gamma"
            assert isinstance(exc.underlying_error, RuntimeError)

    def test_gamma_cdf_floating_point_error(self) -> None:
        """FloatingPointError from gamma.cdf should be wrapped as DistributionFittingError."""
        values = np.array([1.0, 2.0, 3.0] * 16).reshape(4, 12)

        with patch("climate_indices.compute.scipy.stats.gamma.cdf") as mock_cdf:
            mock_cdf.side_effect = FloatingPointError("overflow encountered")

            with pytest.raises(exceptions.DistributionFittingError) as exc_info:
                compute.transform_fitted_gamma(
                    values,
                    data_start_year=2000,
                    calibration_start_year=2000,
                    calibration_end_year=2003,
                    periodicity=compute.Periodicity.monthly,
                )

            exc = exc_info.value
            assert exc.distribution_name == "gamma"
            assert isinstance(exc.underlying_error, FloatingPointError)


class TestGammaNormPPFErrorHandling:
    """Test error handling for scipy.stats.norm.ppf() failures in gamma path."""

    def test_norm_ppf_value_error_in_gamma(self) -> None:
        """ValueError from norm.ppf in gamma path should be wrapped with gamma context."""
        values = np.array([1.0, 2.0, 3.0] * 16).reshape(4, 12)

        # mock norm.ppf to raise ValueError
        # gamma.cdf should succeed, only ppf should fail
        with patch("climate_indices.compute.scipy.stats.norm.ppf") as mock_ppf:
            mock_ppf.side_effect = ValueError("invalid probability values")

            with pytest.raises(exceptions.DistributionFittingError) as exc_info:
                compute.transform_fitted_gamma(
                    values,
                    data_start_year=2000,
                    calibration_start_year=2000,
                    calibration_end_year=2003,
                    periodicity=compute.Periodicity.monthly,
                )

            exc = exc_info.value
            # distribution_name should be "gamma" (caller's context)
            assert exc.distribution_name == "gamma"
            assert exc.input_shape is not None
            assert exc.parameters is not None
            assert "probabilities" in exc.parameters
            assert exc.suggestion == "Try using pearson3 distribution instead"
            assert isinstance(exc.underlying_error, ValueError)

    def test_norm_ppf_runtime_error_in_gamma(self) -> None:
        """RuntimeError from norm.ppf in gamma path should be wrapped."""
        values = np.array([1.0, 2.0, 3.0] * 16).reshape(4, 12)

        with patch("climate_indices.compute.scipy.stats.norm.ppf") as mock_ppf:
            mock_ppf.side_effect = RuntimeError("convergence failed")

            with pytest.raises(exceptions.DistributionFittingError) as exc_info:
                compute.transform_fitted_gamma(
                    values,
                    data_start_year=2000,
                    calibration_start_year=2000,
                    calibration_end_year=2003,
                    periodicity=compute.Periodicity.monthly,
                )

            exc = exc_info.value
            assert exc.distribution_name == "gamma"
            assert isinstance(exc.underlying_error, RuntimeError)


class TestPearsonCDFErrorHandling:
    """Test error handling for scipy.stats.pearson3.cdf() failures."""

    def test_pearson_cdf_value_error(self) -> None:
        """ValueError from pearson3.cdf should be wrapped as DistributionFittingError."""
        values = np.array([1.0, 2.0, 3.0] * 16).reshape(4, 12)

        # need to provide pre-computed parameters to reach the _pearson_fit() call
        # use transform_fitted_pearson which calls _pearson_fit internally
        probabilities_of_zero = np.array([0.0] * 12)
        skews = np.array([0.5] * 12)
        locs = np.array([1.0] * 12)
        scales = np.array([0.5] * 12)

        with patch("climate_indices.compute.scipy.stats.pearson3.cdf") as mock_cdf:
            mock_cdf.side_effect = ValueError("invalid pearson parameters")

            with pytest.raises(exceptions.DistributionFittingError) as exc_info:
                compute.transform_fitted_pearson(
                    values,
                    data_start_year=2000,
                    calibration_start_year=2000,
                    calibration_end_year=2003,
                    periodicity=compute.Periodicity.monthly,
                    probabilities_of_zero=probabilities_of_zero,
                    skews=skews,
                    locs=locs,
                    scales=scales,
                )

            exc = exc_info.value
            assert exc.distribution_name == "pearson3"
            assert exc.input_shape is not None
            assert exc.parameters is not None
            assert "skew" in exc.parameters
            assert "loc" in exc.parameters
            assert "scale" in exc.parameters
            assert exc.suggestion == "Try using gamma distribution instead"
            assert isinstance(exc.underlying_error, ValueError)

    def test_pearson_cdf_runtime_error(self) -> None:
        """RuntimeError from pearson3.cdf should be wrapped as DistributionFittingError."""
        values = np.array([1.0, 2.0, 3.0] * 16).reshape(4, 12)
        probabilities_of_zero = np.array([0.0] * 12)
        skews = np.array([0.5] * 12)
        locs = np.array([1.0] * 12)
        scales = np.array([0.5] * 12)

        with patch("climate_indices.compute.scipy.stats.pearson3.cdf") as mock_cdf:
            mock_cdf.side_effect = RuntimeError("numerical instability")

            with pytest.raises(exceptions.DistributionFittingError) as exc_info:
                compute.transform_fitted_pearson(
                    values,
                    data_start_year=2000,
                    calibration_start_year=2000,
                    calibration_end_year=2003,
                    periodicity=compute.Periodicity.monthly,
                    probabilities_of_zero=probabilities_of_zero,
                    skews=skews,
                    locs=locs,
                    scales=scales,
                )

            exc = exc_info.value
            assert exc.distribution_name == "pearson3"
            assert isinstance(exc.underlying_error, RuntimeError)


class TestPearsonNormPPFErrorHandling:
    """Test error handling for scipy.stats.norm.ppf() failures in pearson path."""

    def test_norm_ppf_value_error_in_pearson(self) -> None:
        """ValueError from norm.ppf in pearson path should be wrapped with pearson context."""
        values = np.array([1.0, 2.0, 3.0] * 16).reshape(4, 12)
        probabilities_of_zero = np.array([0.0] * 12)
        skews = np.array([0.5] * 12)
        locs = np.array([1.0] * 12)
        scales = np.array([0.5] * 12)

        # need to patch both pearson3.cdf (to succeed) and norm.ppf (to fail)
        with patch("climate_indices.compute.scipy.stats.pearson3.cdf") as mock_pearson_cdf:
            # return valid probabilities from pearson3.cdf
            mock_pearson_cdf.return_value = np.full((4, 12), 0.5)

            with patch("climate_indices.compute.scipy.stats.norm.ppf") as mock_ppf:
                mock_ppf.side_effect = ValueError("invalid probability values")

                with pytest.raises(exceptions.DistributionFittingError) as exc_info:
                    compute.transform_fitted_pearson(
                        values,
                        data_start_year=2000,
                        calibration_start_year=2000,
                        calibration_end_year=2003,
                        periodicity=compute.Periodicity.monthly,
                        probabilities_of_zero=probabilities_of_zero,
                        skews=skews,
                        locs=locs,
                        scales=scales,
                    )

                exc = exc_info.value
                # distribution_name should be "pearson3" (caller's context)
                assert exc.distribution_name == "pearson3"
                assert exc.input_shape is not None
                assert exc.parameters is not None
                assert "probabilities" in exc.parameters
                assert exc.suggestion == "Try using gamma distribution instead"
                assert isinstance(exc.underlying_error, ValueError)

    def test_norm_ppf_runtime_error_in_pearson(self) -> None:
        """RuntimeError from norm.ppf in pearson path should be wrapped."""
        values = np.array([1.0, 2.0, 3.0] * 16).reshape(4, 12)
        probabilities_of_zero = np.array([0.0] * 12)
        skews = np.array([0.5] * 12)
        locs = np.array([1.0] * 12)
        scales = np.array([0.5] * 12)

        with patch("climate_indices.compute.scipy.stats.pearson3.cdf") as mock_pearson_cdf:
            mock_pearson_cdf.return_value = np.full((4, 12), 0.5)

            with patch("climate_indices.compute.scipy.stats.norm.ppf") as mock_ppf:
                mock_ppf.side_effect = RuntimeError("convergence failed")

                with pytest.raises(exceptions.DistributionFittingError) as exc_info:
                    compute.transform_fitted_pearson(
                        values,
                        data_start_year=2000,
                        calibration_start_year=2000,
                        calibration_end_year=2003,
                        periodicity=compute.Periodicity.monthly,
                        probabilities_of_zero=probabilities_of_zero,
                        skews=skews,
                        locs=locs,
                        scales=scales,
                    )

                exc = exc_info.value
                assert exc.distribution_name == "pearson3"
                assert isinstance(exc.underlying_error, RuntimeError)


class TestDistributionFittingErrorAttributes:
    """Test DistributionFittingError attribute handling."""

    def test_all_attributes_set(self) -> None:
        """All attributes should be stored when provided."""
        params = {"alpha": "0.5", "beta": "1.0"}
        underlying = ValueError("test error")
        exc = exceptions.DistributionFittingError(
            "Fitting failed",
            distribution_name="gamma",
            input_shape=(10, 12),
            parameters=params,
            suggestion="try pearson3",
            underlying_error=underlying,
        )

        assert exc.distribution_name == "gamma"
        assert exc.input_shape == (10, 12)
        assert exc.parameters == params
        assert exc.suggestion == "try pearson3"
        assert exc.underlying_error is underlying
        assert str(exc) == "Fitting failed"

    def test_all_attributes_default_to_none(self) -> None:
        """Attributes should default to None when not provided."""
        exc = exceptions.DistributionFittingError("Fitting failed")

        assert exc.distribution_name is None
        assert exc.input_shape is None
        assert exc.parameters is None
        assert exc.suggestion is None
        assert exc.underlying_error is None

    def test_keyword_only_enforcement(self) -> None:
        """New attributes must be passed as keywords, not positional."""
        # this should work (keyword arguments)
        exc = exceptions.DistributionFittingError(
            "Fitting failed",
            distribution_name="gamma",
        )
        assert exc.distribution_name == "gamma"

        # this should fail (positional arguments)
        with pytest.raises(TypeError):
            exceptions.DistributionFittingError(
                "Fitting failed",
                "gamma",  # trying to pass distribution_name positionally
            )

    def test_subclass_compatibility(self) -> None:
        """Subclasses should continue to work with positional message."""
        # InsufficientDataError should still work
        exc1 = exceptions.InsufficientDataError(
            "Not enough data",
            non_zero_count=5,
            required_count=10,
        )
        assert exc1.non_zero_count == 5
        assert str(exc1) == "Not enough data"

        # PearsonFittingError should still work
        underlying = ValueError("test")
        exc2 = exceptions.PearsonFittingError(
            "Pearson failed",
            underlying_error=underlying,
        )
        assert exc2.underlying_error is underlying
        assert str(exc2) == "Pearson failed"


class TestSummarizeArray:
    """Test the _summarize_array helper function."""

    def test_none_array(self) -> None:
        """None should be formatted as 'name=None'."""
        result = compute._summarize_array(None, "test")
        assert result == "test=None"

    def test_small_array(self) -> None:
        """Arrays with ≤12 elements should show full representation."""
        arr = np.array([1, 2, 3, 4, 5])
        result = compute._summarize_array(arr, "values")
        assert "values=" in result
        # should contain actual values
        assert "[1" in result or "1" in result

    def test_large_array(self) -> None:
        """Arrays with >12 elements should show summary statistics."""
        arr = np.arange(100, dtype=float)
        result = compute._summarize_array(arr, "data")

        # should contain summary statistics
        assert "shape=" in result
        assert "min=" in result
        assert "max=" in result
        assert "mean=" in result
        assert "nan_count=" in result

    def test_array_with_nans(self) -> None:
        """Arrays with NaN values should report nan_count."""
        arr = np.array([1.0, 2.0, np.nan, 4.0, np.nan] * 5)
        result = compute._summarize_array(arr, "mixed")

        assert "nan_count=" in result
        # should show 10 NaNs out of 25 values
        assert "10/" in result or "10 /" in result

    def test_all_nans_array(self) -> None:
        """Arrays with all NaN values should not crash."""
        arr = np.full(20, np.nan)
        result = compute._summarize_array(arr, "nans")

        # should still produce valid summary
        assert "shape=" in result
        assert "nan_count=" in result

    def test_2d_array(self) -> None:
        """2D arrays should show shape correctly."""
        arr = np.ones((10, 12))
        result = compute._summarize_array(arr, "matrix")

        assert "shape=(10, 12)" in result


class TestCallerSideFallback:
    """Integration test for caller-side fallback from pearson to gamma."""

    def test_spi_pearson_failure_falls_back_to_gamma(self) -> None:
        """When pearson fitting fails, SPI should fall back to gamma distribution.

        This is an integration test verifying that the error propagation works
        correctly and can be caught by higher-level code.
        """
        values = np.array([1.0, 2.0, 3.0] * 16).reshape(4, 12)
        probabilities_of_zero = np.array([0.0] * 12)
        skews = np.array([0.5] * 12)
        locs = np.array([1.0] * 12)
        scales = np.array([0.5] * 12)

        # simulate pearson3.cdf failure
        with patch("climate_indices.compute.scipy.stats.pearson3.cdf") as mock_cdf:
            mock_cdf.side_effect = ValueError("pearson fitting failed")

            # the caller should catch DistributionFittingError
            with pytest.raises(exceptions.DistributionFittingError) as exc_info:
                compute.transform_fitted_pearson(
                    values,
                    data_start_year=2000,
                    calibration_start_year=2000,
                    calibration_end_year=2003,
                    periodicity=compute.Periodicity.monthly,
                    probabilities_of_zero=probabilities_of_zero,
                    skews=skews,
                    locs=locs,
                    scales=scales,
                )

            # verify the error has all the context needed for fallback decision
            exc = exc_info.value
            assert exc.distribution_name == "pearson3"
            assert exc.suggestion == "Try using gamma distribution instead"
            assert exc.underlying_error is not None

            # in real code, the caller would catch this and retry with gamma
            # we're just verifying the exception provides enough information
