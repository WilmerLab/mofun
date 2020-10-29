#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='functionaliseMOF',
      version='0.1.0',
      description='Search for pattern in Metal-organic framework and replace it',
      author='Meiirbek Islamov',
      author_email='mei12@pitt.edu',
      url='https://github.com/meiirbek-islamov/functionaliseMOF.git',
      packages=find_packages(include=['functionaliseMOF']),
      install_requires=[
          'numpy',
          'matplotlib'
      ],
)
