# Copyright (C) 2022 Jae-Won Chung <jwnchung@umich.edu>
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages

setup(
    name="rene",
    version="0.0.1",
    description="A pipeline visualizer for pipeline parallel DNN training",
    long_description="# Rene: A pipeline visualizer for pipeline parallel DNN training\n",
    long_description_content_type="text/markdown",
    url="https://github.com/SymbioticLab/rene",
    author="Jae-Won Chung",
    author_email="jwnchung@umich.edu",
    license="Apache-2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords=["deep-learning", "mlsys", "visualization"],
    packages=find_packages("."),
    install_requires=[
        "matplotlib==3.6.2",
        "networkx==3.0",
        "numpy==1.23.4",
        "scipy==1.10.1",
        "pandas==1.5.3",
    ],
    python_requires=">=3.9",
    extras_require={
        "dev": ["ruff", "black==22.10.0", "mypy==1.1.1", "pytest"],
    }
)
