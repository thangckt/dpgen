"""Some notes
- Must set txt='calc.txt' in GPAW calculator for backward files.
"""

import argparse

from ase.calculators.calculator import PropertyNotImplementedError
from ase.io import Trajectory, read
from ase.parallel import parprint
from gpaw import GPAW, PW, FermiDirac

parprint("GPAW running...")

### parallel args
parallel_args = {
    "sl_auto": True,  # enable ScaLAPACK parallelization
    "use_elpa": True,  # enable Elpa eigensolver
    # "augment_grids":True,  # use all cores for XC/Poisson
    # 'domain': (int(world.size/8), 8, 1),
}

############### ANCHOR: Parameters
### params from command line:
parser = argparse.ArgumentParser(description="Optimize structure using GPAW")
parser.add_argument("--fmax", type=float, default=0.05, help="max force for convergence")
parser.add_argument("--ecutoff", type=float, default=550, help="PW energy cutoff")
parser.add_argument("--kdensity", type=float, default=19, help="k-point density")
parser.add_argument(
    "--pbc", type=str, default="1 1 1", help="periodic boundary condition. E.g., --pbc '1 1 0'"
)
args = parser.parse_args()

fmax = args.fmax
ecut = args.ecutoff
kdensity = args.kdensity
pbc = [int(item) for item in args.pbc.split(" ")]


############### ANCHOR: Define atoms and calculator
### atoms: read POSCAR file
atoms = read("POSCAR", format="vasp")
atoms.set_pbc(pbc)

### Calculator
calc = GPAW(
    mode=PW(ecut),  # planewave ecut  350
    xc="PBE",  # the exchange-correlation functional
    # nbands=20,           # number of bands, automaticaly guese when compute in ground_state
    occupations=FermiDirac(0.01),
    kpts={"density": kdensity, "gamma": True},
    txt="calc_singlepoint.txt",
    # parallel=parallel_args,
)


############### ANCHOR: Calculation
atoms.calc = calc
### compute properties
atoms.get_potential_energy()
atoms.get_forces()

### compute stress only if the calculator supports it
try:
    atoms.get_stress(voigt=False)  # not use voigt notation -> return 3x3 matrix
except PropertyNotImplementedError:
    pass

##### write trajectory (only final frame)
Trajectory("CONF.asetraj", "w").write(atoms)
