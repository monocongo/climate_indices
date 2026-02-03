#!/usr/bin/env python3
"""
Example demonstrating safe floating point comparisons in climate_indices.

This example shows the difference between unsafe direct equality checks
and safe numpy.isclose() comparisons, addressing python:S1244.
"""

import numpy as np

from climate_indices import compute, indices


def demonstrate_floating_point_issues():
    """Show why direct floating point equality is problematic."""

    print("=== Floating Point Comparison Issues ===\n")

    # Example 1: Precision issues with simple arithmetic
    print("1. Basic precision issues:")
    value = 0.1 + 0.1 + 0.1
    print(f"   0.1 + 0.1 + 0.1 = {value}")
    print(f"   value == 0.3: {value == 0.3}")  # False!
    print(f"   np.isclose(value, 0.3): {np.isclose(value, 0.3)}")  # True
    print()

    # Example 2: Statistical parameters from fitting
    print("2. Statistical parameter comparison:")

    # Simulate failed parameter fitting (should return ~0.0)
    failed_params = np.array([0.0, 1e-16, -1e-16, 2e-15])

    # Unsafe way
    unsafe_count = np.sum(failed_params == 0.0)
    print(f"   Direct equality (==): {unsafe_count} zeros found")

    # Safe way
    safe_count = np.sum(np.isclose(failed_params, 0.0, atol=1e-12))
    print(f"   np.isclose method: {safe_count} zeros found")
    print(f"   Parameters: {failed_params}")
    print()

    # Example 3: Real climate data scenario
    print("3. Climate data validation:")

    # Create some test precipitation data
    np.random.seed(42)
    precip_data = np.random.exponential(scale=10.0, size=120)  # 10 years monthly

    # Add some very small "trace" values that should be considered zero
    precip_data[5] = 1e-15
    precip_data[15] = -1e-15  # Numerical noise
    precip_data[25] = 1e-10

    # Count "zero" precipitation events
    direct_zeros = np.sum(precip_data == 0.0)
    safe_zeros = np.sum(np.isclose(precip_data, 0.0, atol=1e-6))

    print(f"   Direct equality: {direct_zeros} zero precipitation events")
    print(f"   Safe comparison: {safe_zeros} zero precipitation events")
    print(f"   Small values: {precip_data[[5, 15, 25]]}")


def demonstrate_test_assertions():
    """Show proper test assertion techniques."""

    print("\n=== Safe Test Assertions ===\n")

    # Generate some SPI values for testing
    np.random.seed(123)
    precip_data = np.random.gamma(shape=2, scale=10, size=60)  # 5 years

    try:
        spi_values = indices.spi(
            values=precip_data,
            scale=3,
            distribution=indices.Distribution.gamma,
            data_start_year=2000,
            calibration_year_initial=2000,
            calibration_year_final=2004,
            periodicity=compute.Periodicity.monthly,
        )

        print("1. Validating SPI computation:")

        # Check that SPI values are reasonable
        valid_spi = spi_values[~np.isnan(spi_values)]

        print(f"   Generated {len(valid_spi)} valid SPI values")
        print(f"   Range: [{np.min(valid_spi):.3f}, {np.max(valid_spi):.3f}]")

        # Safe way to check if any values are close to specific drought thresholds
        severe_drought = np.sum(np.isclose(valid_spi, -2.0, atol=0.1))
        normal_conditions = np.sum(np.isclose(valid_spi, 0.0, atol=0.2))

        print(f"   Values near severe drought (-2.0): {severe_drought}")
        print(f"   Values near normal (0.0): {normal_conditions}")

        # Demonstrate proper test assertion
        expected_range = [-3.5, 3.5]
        assert np.all(valid_spi >= expected_range[0]), "SPI values below expected range"
        assert np.all(valid_spi <= expected_range[1]), "SPI values above expected range"
        print("   ✓ SPI values within expected range")

    except Exception as e:
        print(f"   Error in SPI computation: {e}")


def demonstrate_tolerance_selection():
    """Show how to choose appropriate tolerances for different contexts."""

    print("\n=== Tolerance Selection Guidelines ===\n")

    contexts = {
        "Machine precision": 1e-15,
        "Statistical parameters": 1e-8,
        "Physical measurements": 1e-6,
        "User-facing values": 1e-4,
        "Climate thresholds": 1e-3,
    }

    test_value = 1e-7

    print(f"Testing value: {test_value}")
    print("Tolerance contexts:")

    for context, tolerance in contexts.items():
        is_zero = np.isclose(test_value, 0.0, atol=tolerance)
        print(f"   {context:.<25} (atol={tolerance:>8}): {'Zero' if is_zero else 'Non-zero'}")


if __name__ == "__main__":
    demonstrate_floating_point_issues()
    demonstrate_test_assertions()
    demonstrate_tolerance_selection()

    print("\n=== Summary ===")
    print("✓ Use np.isclose() instead of == for floating point comparisons")
    print("✓ Choose appropriate tolerance (atol) for your context")
    print("✓ Use np.allclose() for array comparisons")
    print("✓ Use np.testing.assert_allclose() in tests")
    print("✗ Avoid direct equality (==) with computed floating point values")
