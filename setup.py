import pathlib
from setuptools import setup

# the directory containing this file
BASE_DIR = pathlib.Path(__file__).parent

# the text of the README file
README = (BASE_DIR / "README.md").read_text()

setup(
    name="climate_indices",
    version="1.0.5",
    url="https://github.com/monocongo/climate_indices",
    license="BSD",
    author="James Adams",
    author_email="monocongo@gmail.com",
    description=(
        "Community reference implementations of climate index "
        "algorithms in Python. Including Palmers (PDSI, scPDSI,  "
        "PHDI, and Z-Index), SPI, SPEI, PET, and PNP."
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    packages=["climate_indices"],
    include_package_data=True,
    install_requires=[
        "dask",
        "nco",
        "netcdf4",
        "numba",
        "numpy",
        "pytest",
        "scipy",
        "toolz",
        "xarray",
    ],
    tests_require=["pytest"],
    test_suite="tests",
    keywords=(
        "indices climate climate_indices drought drought_indices pdsi "
        "spi spei evapotranspiration"
    ),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        'License :: OSI Approved :: BSD License',
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    entry_points={
        "console_scripts": [
            "process_climate_indices=climate_indices.__main__:main",
        ]
    },
)
