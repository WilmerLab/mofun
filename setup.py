#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='mofun',
      version='0.1.1',
      description='Find and replace functional groups in any given periodic structure.',
      author='Paul Boone',
      author_email='narcissus.pong@gmail.com',
      url='https://github.com/wilmerlab/mofun.git',
      packages=find_packages(include=['mofun*']),
      install_requires=[
          'ase',
          'numpy',
          'scipy',
          'PyCifRW',
          'networkx',
          'click',
      ],
      extras_require={
        'docs': ['mkdocs-material', 'mkdocstrings'],
      },
      entry_points={
        'console_scripts': [
            'mofun = mofun.cli.mofun_cli:mofun_cli',
          ]
      }

)
