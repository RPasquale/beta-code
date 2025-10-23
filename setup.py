"""
Setup script for UE-SEA package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ue-sea",
    version="0.1.0",
    author="UE-SEA Team",
    author_email="team@ue-sea.dev",
    description="Unified Evolutionary Software Engineering Agent",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/ue-sea/ue-sea",
    project_urls={
        "Bug Reports": "https://github.com/ue-sea/ue-sea/issues",
        "Source": "https://github.com/ue-sea/ue-sea",
        "Documentation": "https://ue-sea.readthedocs.io",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Code Generators",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "coverage>=7.2.0",
        ],
        "gpu": [
            "cupy-cuda11x>=12.0.0",
            "faiss-gpu>=1.7.4",
        ],
        "distributed": [
            "deepspeed>=0.10.0",
            "accelerate>=0.21.0",
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ue-sea=ue_sea.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="ai, code, evolution, software-engineering, llm, machine-learning",
)
