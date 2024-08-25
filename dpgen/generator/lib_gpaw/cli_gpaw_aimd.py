"""Some notes:
- Run MD in ase following this tutorial: https://wiki.fysik.dtu.dk/ase/tutorials/md/md.html
- Must set txt='calc.txt' in GPAW calculator for backward files.
"""

import argparse
import os

from ase import units
from ase.io import Trajectory, read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.parallel import paropen, parprint
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
parser.add_argument("--nsteps", type=int, default=100, help="number of MD steps")
parser.add_argument("--conf_freq", type=int, default=5, help="timstep interval to save conf")
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

nsteps = args.nsteps
conf_freq = args.conf_freq
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
    txt="calc_aimd.txt",
    parallel=parallel_args,
)

### attach calculator
atoms.calc = calc


############### ANCHOR: Calculation
### Set the momenta corresponding to T=300K
MaxwellBoltzmannDistribution(atoms, temperature_K=100)

### We want to run MD with constant energy using the Langevin algorithm with a time step of 5 fs, the temperature T and the friction coefficient to 0.02 atomic units.
dyn = Langevin(atoms, timestep=0.5 * units.fs, temperature_K=300, friction=0.002)


### tailor properties
def tailor_properties(a=atoms, filename="calc_properties.txt"):
    """Function to print the potential, kinetic and total energy"""
    ### Write the header line
    if not os.path.exists(filename):
        with paropen(filename, "w") as fo:
            fo.write("step epot ekin\n")
    ### Extract properties
    step = dyn.nsteps
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    e = epot + ekin
    ### Append the data to the file
    with paropen(filename, "a") as fo:
        fo.write(f"{step} {e:.7f} {epot:.7f} {ekin:.7f}\n")


dyn.attach(tailor_properties, interval=conf_freq)

### We also want to save the positions of all atoms after every 100th time step.
traj = Trajectory("CONF.asetraj", "w", atoms, properties=["energy", "forces", "stress"])
dyn.attach(traj.write, interval=conf_freq)

### Now run the dynamics
dyn.run(nsteps)

############### ANCHOR: Write final optimized structure
### write poscar
# atoms.write("POSCAR", format="vasp")
