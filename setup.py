from setuptools import setup, find_packages

setup(
    name="neko_utils",
    version="0.1.0",
    description="Utilities for working with Neko datasets using xarray",
    author="L. L. Donati & L. Huusko",
    packages=find_packages(),
    install_requires=[
        "xarray",
        "numpy",
        "pandas",
        "pymech",  
    ],
)
