"""Setup script for the scd-mh package.

SCD-MH: Semantically Constrained Decoding via Metropolis-Hastings.
A production-quality implementation of the algorithms described in:

    "Semantically Constrained Decoding: A Formal Theory of
     Distribution-Aligned Neurosymbolic Generation" (JMLR 2026)
"""

from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="scd-mh",
    version="0.1.0",
    description=(
        "Semantically Constrained Decoding via Metropolis-Hastings: "
        "distribution-aligned neurosymbolic generation for LLMs"
    ),
    long_description=open("README.md", "r").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="[Author Name]",
    author_email="[email]",
    url="https://github.com/placeholder/scd-mh",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "mypy>=1.0",
            "ruff>=0.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="constrained-decoding metropolis-hastings neurosymbolic llm semantic-constraints",
)
