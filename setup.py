from pathlib import Path

from setuptools import setup, find_packages

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name='sedpy_jax',
    version='0.1.1',
    description='JAX-accelerated version of sedpy (filter projections + spectral smoothing) by Ben Johnson',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Amanda Stoffers',
    author_email='aas208@cam.ac.uk',
    url='https://github.com/Espe13/sedpy_jax',
    license='MIT',
    packages=find_packages(where="."),
    include_package_data=True,
    python_requires='>=3.10',
    install_requires=[
        'jax',
        'matplotlib',
        'numpy',
        'scipy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
