#!/usr/bin/env python3
"""
Setup script for XAI Visualization Tool for Histopathology Images.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read version from src/__init__.py
def get_version():
    version_file = os.path.join("src", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

setup(
    name="GRAPHITE",
    version=get_version(),
    author="XAI Visualization Team",
    author_email="contact@example.com",
    description="A comprehensive toolkit for explaining AI model predictions on histopathology images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raktim-mondol/GRAPHITE",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9.2",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "GRAPHITE=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml"],
    },
    keywords="explainable-ai, histopathology, medical-imaging, deep-learning, visualization",
    project_urls={        "Bug Reports": "https://github.com/raktim-mondol/GRAPHITE/issues",
        "Source": "https://github.com/raktim-mondol/GRAPHITE",
        "Documentation": "https://xai.readthedocs.io/",
    },
) 