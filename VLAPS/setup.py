from setuptools import setup, find_packages

setup(
    name="vlaps",                    # Project name
    version="0.1",                        # Initial version
    packages=find_packages(),            # Automatically find all packages
    install_requires=[],                 # List of runtime dependencies
    python_requires=">=3.7",             # Minimum Python version
)