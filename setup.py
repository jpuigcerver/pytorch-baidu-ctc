import io
import os
import sys

import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

extra_compile_args = {
    "cxx": ["-std=c++11", "-O3", "-fopenmp"],
    "nvcc": ["-std=c++11", "-O3", "--compiler-options=-fopenmp"],
}

CC = os.getenv("CC", None)
if CC is not None:
    extra_compile_args["nvcc"].append("-ccbin=" + CC)

include_dirs = [
    "{}/third-party/warp-ctc/include".format(
        os.path.dirname(os.path.realpath(__file__))
    )
]

sources = ["src/binding.cc"]

if torch.cuda.is_available():
    sources += [
        "third-party/warp-ctc/src/ctc_entrypoint.cu",
        "third-party/warp-ctc/src/reduce.cu",
    ]

    Extension = CUDAExtension
else:
    sources += ["third-party/warp-ctc/src/ctc_entrypoint.cpp"]

    Extension = CppExtension


setup(
    name="torch-baidu-ctc",
    version="0.1",
    description="PyTorch bindings for Baidu Warp-CTC",
    long_description=io.open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jpuigcerver/pytorch-baidu-ctc",
    author="Joan Puigcerver",
    author_email="joapuipe@gmail.com",
    license="Apache",
    packages=find_packages(),
    ext_modules=[
        Extension(
            name="_torch_baidu_ctc",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    setup_requires=["pybind11", "torch>=0.4.1"],
    install_requires=["pybind11", "torch>=0.4.1"],
)
