# MOFUN

[![MOFUN Refactor Branch Test Status](https://github.com/wilmerlab/mofun/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/wilmerlab/mofun/actions/workflows/python-app.yml)
[![pypi version](https://img.shields.io/pypi/v/mofun.svg)](https://pypi.org/project/mofun/)

MOFUN is a Python package that can find and replace patterns in a molecular structure, across periodic boundaries.

## Installation

Requires Python > 3.8.

```
pip install mofun
```

## Examples

```
mofun {input-structure} {output-structure} -f {find-pattern} -r {replace-pattern}
```

See [Documentation](https://wilmerlab.github.io/mofun/)
