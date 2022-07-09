# compile command : python setup.py build_ext -i
import os, sys

from distutils.core import setup, Extension
from distutils import sysconfig

cpp_args = ['-std=c++14']

ext_modules = [
    Extension(
    'WLsubtree',
        ['WLsubtree.cpp'],
        include_dirs=['pybind11/include'],
    language='c++',
    extra_compile_args = cpp_args,
    ),
]

setup(
    name='WLsubtree',
    version='0.0.2',
    description='Class to build Weisfeiler-Lehman subtree',
    ext_modules=ext_modules,
)