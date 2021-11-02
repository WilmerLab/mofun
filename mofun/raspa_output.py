import re

def parse_gas_loading(output_file, units='molkg'):
    atom_blocks = []

    with open(output_file) as origin:
        lines = origin.read().split('\n')
        absvloading = None
        absvloading_error = None
        for i, line in enumerate(lines):
            if units=='molkg' and "Average loading absolute [mol/kg framework]" in line:
                absvloading = float(line.split()[5])
                absvloading_error = float(line.split()[7])
            elif units=='vv' and "Average loading absolute [cm^3 (STP)/cm^3 framework]" in line:
                absvloading = float(line.split()[6])
                absvloading_error = float(line.split()[8])
            elif units=='cc_g' and "Average loading absolute [cm^3 (STP)/gr framework]" in line:
                absvloading = float(line.split()[6])
                absvloading_error = float(line.split()[8])
            elif units=='molkg' and "Average loading excess [mol/kg framework]" in line:
                excessvloading = float(line.split()[5])
                excessvloading_error = float(line.split()[7])
            elif units=='vv' and "Average loading excess [cm^3 (STP)/cm^3 framework]" in line:
                excessvloading = float(line.split()[6])
                excessvloading_error = float(line.split()[8])
            elif units=='cc_g' and "Average loading excess [cm^3 (STP)/gr framework]" in line:
                excessvloading = float(line.split()[6])
                excessvloading_error = float(line.split()[8])
            elif "Number of molecules:" in line:
                atom_blocks = [float(lines[offset + i + 5].split()[2]) for offset in range(5)]
            elif "Conversion factor molecules/unit cell -> cm^3 STP/cm^3:" in line:
                molecules_uc_to_vv = float(line.split()[7])
            elif "Conversion factor mol/kg -> cm^3 STP/cm^3:" in line:
                molkg_to_vv = float(line.split()[6])

    return absvloading, absvloading_error, excessvloading, excessvloading_error, *atom_blocks, molecules_uc_to_vv, molkg_to_vv


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
            elif "Conversion factor molecules/unit cell -> cm^3 STP/cm^3:" in line:
                molecules_uc_to_vv = float(line.split()[7])
            elif "Conversion factor mol/kg -> cm^3 STP/cm^3:" in line:
                molkg_to_vv = float(line.split()[6])
    return gas_henrys_error, molecules_uc_to_vv, molkg_to_vv
