from setuptools import setup, find_packages

setup(
    name="legacycodebench",
    version="1.0.0",
    description="Benchmark for AI systems on legacy code understanding and documentation",
    author="Kalmantic Applied AI Lab",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "gitpython>=3.1.40",
        "jsonschema>=4.20.0",
        "networkx>=3.2.1",
        "nltk>=3.8.1",
        "markdown>=3.5.1",
        "beautifulsoup4>=4.12.2",
        "openai>=1.12.0",
        "anthropic>=0.18.0",
        "boto3>=1.34.0",
        "botocore>=1.34.0",
        "click>=8.1.7",
        "tqdm>=4.66.1",
        "pandas>=2.1.4",
    ],
    entry_points={
        "console_scripts": [
            "legacycodebench=legacycodebench.cli:main",
        ],
    },
    python_requires=">=3.8",
)

