import json
from pathlib import Path

from setuptools import find_packages, setup

def _requirements():
    return Path("requirements.txt").read_text()

setup(
    name="faster-chatglm-6b",
    version='0.0.0',
    description="",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    install_requires=_requirements(),
    entry_points={},
    packages=find_packages(),
    url="",
    author="",
    scripts={},
    include_package_data=True,
    python_requires=">=3.7",
    license=""
)
