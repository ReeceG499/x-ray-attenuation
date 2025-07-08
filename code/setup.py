from setuptools import setup, find_packages

setup(
    name="ct_simulation",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
)