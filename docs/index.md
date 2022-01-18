# MOFUN

## Installation

Requires Python > 3.8.

```
pip install mofun
```

## Command-line usage:

```
mofun {input-structure} {output-structure} -f {find-pattern} -r {replace-pattern}
```

```
mofun --help                                                                                                                                                                                                                                                                                    Tue Jan 18 12:00:59 2022
Usage: mofun [OPTIONS] INPUTPATH OUTPUTPATH

Options:
  -f, --find PATH
  -r, --replace PATH
  -p, --replace-fraction FLOAT
  --atol FLOAT                  absolute tolerance in Angstroms for atom
                                posistions to be considered matching
  -ap1, --axisp1-idx INTEGER    index of first point on primary rotation axis
  -ap2, --axisp2-idx INTEGER    index of second point on primary rotation axis
  -op, --opoint-idx INTEGER     index of point that makes up secondary
                                rotation axis (between this point and the
                                primary rotation axis)
  --dumppath PATH
  -q, --chargefile FILENAME
  --replicate INTEGER...        replicate structure across x, y, and z
                                dimensions
  --mic FLOAT                   enforce minimum image convention using a
                                cutoff of mic
  --framework-element TEXT      convert all atoms that are in group 0, the
                                framework group to a specific atom type to
                                make vizualizing the structure easier
  --pp                          Assign UFF pair potentials to atoms
                                (sufficient for fixed force-field
                                calculations)
  --help                        Show this message and exit.
```

## Pyrhon documentation:

See [Atoms](reference/atoms.md) for info on Atoms objects and how to use them.

See [mofun](reference/mofun.md) for find_pattern_in_structure() and replace_pattern_in_structure().

See [Examples](examples.md) for full examples on real systems, using both the command line and the Python interfaces.


