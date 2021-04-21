import re

def parse_gas_loading(output_file):
    atom_blocks = []

    with open(output_file) as origin:
        lines = origin.read().split('\n')
        for i, line in enumerate(lines):
            if "Average loading absolute [mol/kg framework]" in line:
                absvloading = float(line.split()[5])
                absvloading_error = float(line.split()[7])
            # if "Average loading absolute [cm^3 (STP)/cm^3 framework]" in line:
                # absvloading = float(line.split()[6])
                # absvloading_error = float(line.split()[8])
            elif "Number of molecules:" in line:
                atom_blocks = [float(lines[offset + i + 5].split()[2]) for offset in range(5)]
            elif "Conversion factor molecules/unit cell -> cm^3 STP/cm^3:" in line:
                atoms_uc_to_vv = float(line.split()[7])

    return absvloading, absvloading_error, *atom_blocks, atoms_uc_to_vv


def parse_henrys(output_file):
    fl = r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?"

    # captured groups are [gas, Henry's coefficient, Henry's coefficient error]
    henrys_re = re.compile(r"\s+\[([-\w]+)\] Average Henry coefficient:  ({fl}) \+\/\- ({fl}) \[mol/kg/Pa\]".format(fl=fl))

    gas_henrys_error = []
    with open(output_file) as f:
        for line in f:
            m=re.match(henrys_re, line)
            if m:
                matches = list(m.groups())
                matches[1] = float(matches[1])
                matches[2] = float(matches[2])
                gas_henrys_error += [matches]
    return gas_henrys_error
