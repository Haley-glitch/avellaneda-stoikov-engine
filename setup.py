from setuptools import setup, find_packages

setup(
    name="avellaneda-stoikov-engine",
    version="1.0.0",
    author="Your Name",
    description="End-to-end Avellaneda-Stoikov Market Making Engine with MLE Calibration and Backtest",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "plotly>=5.14",
        "statsmodels>=0.14",
        "tqdm>=4.65",
        "pyyaml>=6.0",
        "jinja2>=3.1",
        "numba>=0.57",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3",
            "pytest-cov>=4.1",
            "black>=23.3",
            "isort>=5.12",
            "mypy>=1.3",
            "jupyter>=1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "as-pipeline=run_pipeline:main",
        ]
    },
)
