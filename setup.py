from setuptools import setup, find_packages

setup(
    name="senlytics",
    version="0.0.1",
    description="Tools for fetching and analyzing low-cost air quality sensor data",
    author="Congbo Song",
    author_email="congbo.song@ncas.ac.uk",
    url="https://github.com/dsncas/senlytics",
    packages=find_packages(),
    install_requires=[
        "pandas", "requests", "tqdm", "numpy", "matplotlib",
        "scipy", "statsmodels", "hvplot", "holoviews"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
