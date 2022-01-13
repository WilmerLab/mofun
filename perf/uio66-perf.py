"""
Run and visualize with:

python -m cProfile -o uio66.pstats  -s tottime uio66-perf.py
snakeviz uio66.pstats

"""
import time

from mofun import Atoms, replace_pattern_in_structure

repl = 20

uio66 = Atoms.load("uio-66-67-perf/uio66.cif")
uio66_linker = Atoms.load("uio-66-67-perf/uio66-linker.cml")
structure = uio66.replicate(repldims=(repl, repl, repl))

time_s = time.process_time()
perf_s = time.perf_counter()
result, num_matches = replace_pattern_in_structure(structure, uio66_linker, uio66_linker, return_num_matches=True)
print(time.process_time() - time_s)
print(num_matches)
assert num_matches == 24 * repl**3


