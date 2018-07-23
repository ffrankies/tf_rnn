"""Setup script for the tf_rnn module
"""
import setuptools

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name="tf_rnn",
    version="0.6.0",
    author="ffrankies",
    author_email="wanyef@mail.gvsu.edu",
    description="Tensorflow implementation of an RNN for generating sequences",
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
