#!/usr/bin/env python

from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='mofun',
      version='0.9',
      description='Find and replace functional groups in any given periodic structure.',
      long_description = long_description,
      long_description_content_type = "text/markdown",
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
          'ordered-set'
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
