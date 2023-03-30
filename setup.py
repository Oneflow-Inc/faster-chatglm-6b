from setuptools import find_packages, setup


def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


setup(
    name="faster-chatglm-6b",
    version="0.0.0",
    description="faster-chatglm-6b is a project that uses the OneFlow as the backend to accelerate THUDM/chatglm-6b.",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    install_requires=read_file("requirements.txt").splitlines(),
    entry_points={},
    packages=find_packages(),
    url="https://github.com/Oneflow-Inc/faster-chatglm-6b",
    author="OneFlow-Inc",
    scripts={},
    include_package_data=True,
    python_requires=">=3.7",
    license="Apache License, Version 2.0",
)
