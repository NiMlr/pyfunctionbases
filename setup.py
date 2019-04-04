import os.path
from setuptools import setup

here = os.path.dirname(__file__)
readme_path = os.path.join(here, 'README.md')
readme = open(readme_path, 'rb').read().decode('utf-8')

setup(name='pyfunctionbases',
      use_scm_version={
          'version_scheme': 'post-release',
          'local_scheme': 'dirty-tag'
      },
      description='A Python module to compute multidimensional arrays of evaluated functions.',
      long_description=readme,
      url='https://github.com/NiMlr/PyFunctionBases',
      author='Nils Mueller',
      author_email='nils.mueller@ini.rub.de',
      license='MIT',
      packages=['pyfunctionbases'],
      setup_requires=['setuptools_scm >= 1.7.0'])
