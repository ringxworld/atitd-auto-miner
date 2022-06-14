#!/usr/bin/env python

from setuptools import find_packages
from distutils.core import setup

version = {}

with open('requirements.txt', 'r') as fp:
    required = fp.read().splitlines()

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(name='AtitdScripts',
      version="0.0.1",
      description='',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='',
      author_email='',
      url='',
      packages=find_packages(),
      install_requires=required,
      python_requires='>=3.6',
      entry_points={
          'console_scripts': [
              'run_miner=AtitdScripts.cli.run_mining:main',
              'run_foreground_alert=AtitdScripts.cli.run_foreground_alert:main',
              'run_autowalker=AtitdScripts.cli.run_web_walker:main'
          ]
      }
     )