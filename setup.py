"""Setup script for the tf_rnn module
"""
import setuptools

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name="tf_rnn",
    version="0.5.0",
    author="ffrankies",
    author_email="hahanotsharing",
    description="wanyef@mail.gvsu.edu",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/ffrankies/tf_rnn",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Ubuntu",
    ),
)
