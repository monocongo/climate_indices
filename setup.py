from setuptools import setup

setup(
    name='climate_indices',
    version='1.0.0',
    url='https://github.com/monocongo/climate_indices',
    license='Unlicense',
    author='James Adams',
    author_email='monocongo@gmail.com',
    description=('Community reference implementations of climate index '
                 'algorithms in Python. Including Palmers (PDSI, scPDSI,  '
                 'PHDI, and Z-Index), SPI, SPEI, PET, and PNP.'),
    packages=['climate_indices'],
    install_requires=[
        "netcdf4",
        "numba",
        "numpy",
        "pandas",
        "scipy",
    ],
    tests_require=[
        "nose",
    ],
    test_suite='tests',
    keywords=('indices climate climate_indices drought drought_indices pdsi '
              'spi spei evapotranspiration'),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: Public Domain',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)
