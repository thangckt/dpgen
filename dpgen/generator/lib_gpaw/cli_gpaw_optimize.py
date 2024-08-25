"""Some notes
- Must set txt='calc.txt' in GPAW calculator for backward files.
"""

import argparse

from ase.filters import FrechetCellFilter
from ase.io import Trajectory, read
from ase.optimize import BFGS
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
parser.add_argument(
    "--relax_dim",
    type=str,
    default="1 1 1 0 0 0",
    help="box dimension to relax. E.g., --relax_dim '1 1 1 0 0 0'",
)
args = parser.parse_args()

fmax = args.fmax
ecut = args.ecutoff
kdensity = args.kdensity
pbc = [int(item) for item in args.pbc.split(" ")]
relax_dim = [int(item) for item in args.relax_dim.split(" ")]


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
    txt="calc_optimize.txt",
    parallel=parallel_args,
)

### attach calculator
atoms.calc = calc

############### ANCHOR: Relax structure
atoms_filter = FrechetCellFilter(atoms, mask=relax_dim)
opt = BFGS(atoms_filter)


### write trajectory (only final frame)
Trajectory("CONF.asetraj", "w").write(atoms)
