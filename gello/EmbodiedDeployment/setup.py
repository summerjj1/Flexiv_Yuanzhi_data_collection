from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)
from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

requires = []
with open("envs/pip/req_install.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line.startswith("#"):
            requires.append(line)

setup(
    name="xdeploy",
    version="0.0.2",
    author="PJLab-EmbodiedAI",
    author_email="wangbolun@pjlab.org.cn;zhuyangkun@pjlab.org.cn;wangjiaheng@pjlab.org.cn",
    description="One-step deployment framework for embodied intelligence",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=[x for x in find_packages(".") if x.startswith("xdeploy")],
    include_package_data=True,
    package_data={"xdeploy": ["assets/*"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=requires,
    python_requires=">=3.8",
)
