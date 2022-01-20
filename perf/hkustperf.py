"""
Run and visualize with:

python -m cProfile -o hkustperf.pstats  -s tottime hkustperf.py
snakeviz hkustperf.pstats

"""


import ase
import ase.io

import numpy as np

from mofun import find_pattern_in_structure, replace_pattern_in_structure, Atoms


hkust1_cif = Atoms.load_p1_cif("../tests/hkust-1/hkust-1-with-bonds.cif")
benzene = Atoms.from_ase_atoms(ase.io.read("../tests/molecules/benzene.xyz"))
hkust1_repped = hkust1_cif.replicate(repldims=(3,3,3))
match_indices = find_pattern_in_structure(hkust1_repped, benzene)
