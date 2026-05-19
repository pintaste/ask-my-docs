from setuptools import setup, find_packages
setup(
    name="ask-my-docs",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "anthropic>=0.40.0",
        "sentence-transformers>=3.0.0",
        "faiss-cpu>=1.8.0",
        "pdfplumber>=0.11.0",
        "click>=8.1.0",
        "numpy>=1.26.0",
    ],
    entry_points={"console_scripts": ["ask=ask:cli"]},
)
