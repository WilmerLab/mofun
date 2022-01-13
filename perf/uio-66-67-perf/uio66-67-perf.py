
import csv
import sys
import time

import numpy as np

from mofun import Atoms, replace_pattern_in_structure, find_pattern_in_structure

# all_repldims = [(1,1,1), (2,2,2)]
# all_repldims = [(1,1,1), (2,2,2), (4,4,4)]
all_repldims = [(1,1,1), (2,2,2), (4,4,4), (5,5,5), (6,6,6), (7,7,7), (8,8,8)]

uio66 = Atoms.load("uio66.cif")
uio66_linker = Atoms.load("uio66-linker.cml")


output_csv = csv.writer(open("perf.csv", "w"))
output_csv.writerow(["mof", "repl", "operation", "num_atoms", "matches_found", "matches_expected", "process_time-seconds", "perf_counter_seconds"])


for repldims in all_repldims:
    print("find: ", repldims, )
    structure = uio66.replicate(repldims=repldims)
    time_s = time.process_time()
    perf_s = time.perf_counter()
    patterns = find_pattern_in_structure(structure, uio66_linker)
    output_csv.writerow(("uio66", "%dx%dx%dx" % repldims, "find", len(structure), len(patterns), 24 * repldims[0] * repldims[1] * repldims[2], time.process_time() - time_s, time.perf_counter() - perf_s))
    print(time.process_time() - time_s)

for repldims in all_repldims:
    print("replace: ", repldims)
    structure = uio66.replicate(repldims=repldims)
    time_s = time.process_time()
    perf_s = time.perf_counter()
    patterns = replace_pattern_in_structure(structure, uio66_linker, uio66_linker)
    output_csv.writerow(("uio66", "%dx%dx%dx" % repldims, "replace", len(structure), len(patterns), 24 * repldims[0] * repldims[1] * repldims[2], time.process_time() - time_s, time.perf_counter() - perf_s))
    print(time.process_time() - time_s)

## UIO-67

# uio67 = Atoms.load("uio67.cif")
# uio67_linker = Atoms.load("uio67-linker.cml")


# for repldims in all_repldims:
#     structure = uio67.replicate(repldims=repldims)
#     time_s = time.process_time()
#     perf_s = time.perf_counter()
#     patterns = find_pattern_in_structure(structure, uio67_linker)
#     output_csv.writerow(("uio67", "%dx%dx%dx" % repldims, len(structure), len(patterns), 24 * repldims[0] * repldims[1] * repldims[2], time.process_time() - time_s, time.perf_counter() - perf_s))

# for repldims in all_repldims:
#     structure = uio67.replicate(repldims=repldims)
#     time_s = time.process_time()
#     perf_s = time.perf_counter()
#     patterns = replace_pattern_in_structure(structure, uio67_linker, uio67_linker)
#     output_csv.writerow(("uio67", "%dx%dx%dx" % repldims, len(structure), len(patterns), time.process_time() - time_s, time.perf_counter() - perf_s))





