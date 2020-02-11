import pathlib
from setuptools import find_packages, setup

# the directory containing this file
BASE_DIR = pathlib.Path(__file__).parent

# the text of the README file
README = (BASE_DIR / "README.md").read_text()

setup(
    name="climate_indices",
    version="1.0.9",
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
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=[
        "dask",
        "nco",
        "netcdf4",
        "numba",
        "numpy",
        "scipy",
        "toolz",
        "xarray",
    ],
    keywords=[
        "indices", "climate", "climate indices", "drought",
        "drought indices", "pdsi ", "spi", "spei", "evapotranspiration",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        'License :: OSI Approved :: BSD License',
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    entry_points={
        "console_scripts": [
            "process_climate_indices=climate_indices.__main__:main",
            "spi=climate_indices.__spi__:main",
        ]
    },
)
