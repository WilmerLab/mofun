#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='mofun',
      version='0.1.0',
      description='Find and replace functional groups in any given periodic structure.',
      author='Paul Boone and Meiirbek Islamov',
      author_email='paulboone@pitt.edu mei12@pitt.edu',
      url='https://github.com/wilmerlab/mofun.git',
      packages=find_packages(include=['mofun']),
      install_requires=[
          'ase'
      ],
)
