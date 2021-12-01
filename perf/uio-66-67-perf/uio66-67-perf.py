
import csv
import sys
import time


import numpy as np

from mofun import Atoms, replace_pattern_in_structure, find_pattern_in_structure

all_repldims = [(1,1,1), (2,2,2)]
# all_repldims = [(1,1,1), (2,2,2), (4,4,4)]
# all_repldims = [(1,1,1), (2,2,2), (4,4,4), (8,8,8)]

uio66 = Atoms.load("uio66.cif")
uio67 = Atoms.load("uio67.cif")

uio66_linker = Atoms.load("uio66-linker.cml")
uio67_linker = Atoms.load("uio67-linker.cml")

output_csv = csv.writer(sys.stdout)
output_csv.writerow(["mof", "repl", "num-atoms", "matches-found", "matches-expected", "process-time-seconds", "perf-counter-seconds"])

for repldims in all_repldims:
    structure = uio66.replicate(repldims=repldims)
    time_s = time.process_time()
    perf_s = time.perf_counter()
    patterns = find_pattern_in_structure(structure, uio66_linker)
    output_csv.writerow(("uio66", "%dx%dx%dx" % repldims, len(structure), len(patterns), 24 * repldims[0] * repldims[1] * repldims[2], time.process_time() - time_s, time.perf_counter() - perf_s))

for repldims in all_repldims:
    structure = uio67.replicate(repldims=repldims)
    time_s = time.process_time()
    perf_s = time.perf_counter()
    patterns = find_pattern_in_structure(structure, uio67_linker)
    output_csv.writerow(("uio67", "%dx%dx%dx" % repldims, len(structure), len(patterns), 24 * repldims[0] * repldims[1] * repldims[2], time.process_time() - time_s, time.perf_counter() - perf_s))

for repldims in all_repldims:
    structure = uio66.replicate(repldims=repldims)
    time_s = time.process_time()
    perf_s = time.perf_counter()
    patterns = replace_pattern_in_structure(structure, uio66_linker, uio66_linker)
    output_csv.writerow(("uio66", "%dx%dx%dx" % repldims, len(structure), len(patterns), time.process_time() - time_s, time.perf_counter() - perf_s))

for repldims in all_repldims:
    structure = uio67.replicate(repldims=repldims)
    time_s = time.process_time()
    perf_s = time.perf_counter()
    patterns = replace_pattern_in_structure(structure, uio67_linker, uio67_linker)
    output_csv.writerow(("uio67", "%dx%dx%dx" % repldims, len(structure), len(patterns), time.process_time() - time_s, time.perf_counter() - perf_s))





