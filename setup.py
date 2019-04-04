import os.path
from setuptools import setup

long_description = open('./README.md').read()

setup(name='pyfunctionbases',
      version="1.0",
      description='A Python module to compute multidimensional arrays of evaluated functions.',
      long_description=long_description,
      url='https://github.com/NiMlr/PyFunctionBases',
      author='Nils Mueller',
      author_email='nils.mueller@ini.rub.de',
      license='MIT',
      packages=['pyfunctionbases'],
      setup_requires=['setuptools_scm >= 1.7.0'])
