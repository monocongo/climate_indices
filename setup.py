from setuptools import setup, find_packages

setup(
    name='indices_python',
    version='0.1',
    url='https://github.com/monocongo/indices_python',
    license='GPL 2.0',
    author='James Adams',
    author_email='james.adams@noaa.gov',
    description='Community reference implementations of climate indices algorithms in Python. Including Palmers (PDSI, scPDSI,  PHDI, and Z-Index), SPI, SPEI, PET, and PNP.',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    keywords="indices climate climate_indices drought drought_indices pdsi spi spei evapotranspiration",
)
