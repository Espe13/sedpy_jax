from setuptools import setup, find_packages

setup(
    name='sedpy_jax',
    version='0.1.0',
    description='JAX-accelerated version of SEDpy by Ben Johnson',
    author='Amanda Stoffers',
    packages=find_packages(where="."), 
    include_package_data=True,
    install_requires=[
        'jax',
        'matplotlib',
        'numpy',
        'scipy',
    ],
        classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)