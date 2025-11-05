from setuptools import setup

setup(
    name="wm",
    version="0.1.0",
    py_modules=["model", "main"],
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "datasets>=2.12.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",
        "peft>=0.5.0",
    ],
)

