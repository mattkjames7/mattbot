#!/usr/bin/env python
"""Setup script for agent-oss package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="agent-oss",
    version="0.1.0",
    author="Matt",
    description="An AI agent system with tool-calling capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/USERNAME/REPO",
    packages=find_packages(exclude=["tests*", "env*", ".github*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.12",
    install_requires=[
        "requests>=2.31.0",
        "numpy>=1.24.0",
        "ddgs>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    keywords="ai agent llm tools ollama copilot",
    project_urls={
        "Documentation": "https://github.com/USERNAME/REPO#readme",
        "Source": "https://github.com/USERNAME/REPO",
        "Bug Reports": "https://github.com/USERNAME/REPO/issues",
    },
)
