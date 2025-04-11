from setuptools import setup, find_packages

setup(
    name='sedpy_jax',
    version='0.1.0',
    description='JAX-optimized SED fitting tools inspired by sedpy',
    author='Ben Johnson, jaxified by Amanda Stoffers',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'jax',
        'matplotlib',
        'numpy',
        'scipy',
        # optionally rich, astropy, etc
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)