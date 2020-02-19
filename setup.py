#!/usr/bin/env python

from setuptools import setup

# setup(name='too_short',
#       version='1.0',
#       # list folders, not files
#       packages=['too_short',
#                 'too_short.test'],
#       scripts=['capitalize/bin/cap_script.py'],
#       package_data={'capitalize': ['data/cap_data.txt']},
#       )
with open("README", 'r') as f:
    long_description = f.read()
setup(
    name='too_short',
    version='1.0',
    long_description=long_description,
    description='Module to simplify supervised ml work with sklearn',
    author='Elliott Ribner',
    packages=['too_short'],  # same as name # list folders, not files
    install_requires=['sklearn', 'pandas', 'numpy', 'imblearn',
                      'unittest', 'collections'],  # external packages as dependencies
)
