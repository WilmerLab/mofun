"""
Take CIF file, optionally update with charges from chargefile, replicate to meet MIC, assign pair
potentials and output to a working LAMMPS Data file.
"""

import pathlib

import ase
import click
import numpy as np
from mofun import Atoms, replace_pattern_in_structure, find_pattern_in_structure
from mofun.rough_uff import pair_coeffs
from mofun.uff4mof import uff_key_starts_with

@click.command()
@click.argument('inputpath', type=click.Path(path_type=pathlib.Path))
@click.argument('outputpath', type=click.Path(path_type=pathlib.Path))
@click.option('-f', '--find', 'find_path', type=click.Path(path_type=pathlib.Path))
@click.option('-r', '--replace', 'replace_path', type=click.Path(path_type=pathlib.Path))
@click.option('-p', '--replace-fraction', type=float, default=1.0)
@click.option('--atol', type=float, default=5e-2, help="absolute tolerance in Angstroms for atom posistions to be considered matching")
@click.option('-ap1', '--axisp1-idx', type=int, default=None, help="index of first point on primary rotation axis")
@click.option('-ap2', '--axisp2-idx', type=int, default=None, help="index of second point on primary rotation axis")
@click.option('-op', '--opoint-idx', type=int, default=None, help="index of point that makes up secondary rotation axis (between this point and the primary rotation axis)")
@click.option('--dumppath', type=click.Path(path_type=pathlib.Path))
@click.option('-q', '--chargefile', type=click.File('r'))
@click.option('--replicate', nargs=3, type=int, help="replicate structure across x, y, and z dimensions")
@click.option('--mic', type=float, help="enforce minimum image convention using a cutoff of mic")
@click.option('--framework-element', type=str, help="convert all atoms that are in group 0, the framework group to a specific atom type to make vizualizing the structure easier")
@click.option('--pp', is_flag=True, default=False, help="Assign UFF pair potentials to atoms (sufficient for fixed force-field calculations)")
def mofun_cli(inputpath, outputpath,
        find_path=None, replace_path=None, atol=5e-2, replace_fraction=1.0, axisp1_idx=None, axisp2_idx=None, opoint_idx=None,
        dumppath=None, chargefile=None, replicate=None, mic=None, framework_element=None, pp=False):
    atoms = Atoms.load(inputpath)

    # upate positions from lammps dump file
    if dumppath is not None:
        # update positions in original atoms file with new positions
        dumpatoms = ase.io.read(dumppath, format="lammps-dump-text")
        assert len(dumpatoms.positions) == len(atoms.positions)
        atoms.positions = dumpatoms.positions

    # update charges
    if chargefile is not None:
        charges = np.array([float(line.strip()) for line in chargefile if line.strip() != ''])
        assert len(charges) == len(atoms.positions)
        atoms.charges = charges

    if replicate is not None:
        atoms = atoms.replicate(replicate)

    # replicate to meet minimum image convention, if necessary
    if mic is not None:
        if atoms.cell_is_orthorhombic():
            repls = np.array(np.ceil(2*mic / np.diag(atoms.cell)), dtype=int)
            atoms = atoms.replicate(repls)
        else:
            print ("WARNING: Minimimum image convention is only implemented for orthorhombic structures, please use --replicate")

    if pp:
        assign_pair_params_to_structure(atoms)

    if replace_path is not None and find_path is None:
        print("Cannot perform a replace operation without a find operation")
    elif find_path is not None:
        search_pattern = Atoms.load(find_path)
        if replace_path is not None:
            replace_pattern = Atoms.load(replace_path)
            atoms = replace_pattern_in_structure(atoms, search_pattern, replace_pattern, atol=atol,
                axisp1_idx=axisp1_idx, axisp2_idx=axisp2_idx, opoint_idx=opoint_idx, replace_fraction=replace_fraction)
        else:
            results = find_pattern_in_structure(atoms, search_pattern, atol=atol)
            print("Found %d instances of the search_pattern in the structure" % len(results))
            print(results)

    # set framework elements to specified element-only works on ASE exports
    if framework_element is not None:
        atoms.symbols[atoms.atom_groups == 0] = framework_element

    if outputpath.suffix in ['.lmpdat', '.mol', '.cif']:
        atoms.save(outputpath)
    else:
        print("INFO: Trying output using ASE")
        aseatoms = atoms.to_ase()
        if framework_element is not None:
            aseatoms.symbols[atoms.atom_groups == 0] = framework_element

        aseatoms.set_pbc(True)
        aseatoms.write(outputpath)

def assign_pair_params_to_structure(structure):
    # NOTE: in UFF, pair params should always be the same for atoms of the same element, regardless of type
    # DUPLICATE in functionalize_linkers.py: should be refactored
    uff_keys = [uff_key_starts_with(el.ljust(2, "_"))[0] for el in structure.atom_type_elements]
    structure.pair_coeffs = ['%10.6f %10.6f # %s' % (*pair_coeffs(k), k) for k in uff_keys]
    structure.atom_type_labels = uff_keys
