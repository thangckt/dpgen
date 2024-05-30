"""
NOTE: do not use `return` in the functions that run dpdispatcher.submission
"""

import glob
import os
import shutil
import subprocess as sp
import sys

import dpdata
from ase.io import Trajectory
from ase.io.vasp import write_vasp
from packaging.version import Version

from dpgen.dispatcher.Dispatcher import make_submission
from dpgen.generator.lib.utils import symlink_user_forward_files, check_api_version

### use from...import... may cause circular import. To avoid this, functions in `gen` file must be defined before importing `gpaw_init`
from ..gen import (
    create_path,
    global_dirname_02,
    global_dirname_03,
    global_dirname_04,
    poscar_shuffle,
)

# global_dirname_02 = "00.place_ele"
# global_dirname_03 = "01.scale_pert"
# global_dirname_04 = "02.md"


##### ANCHOR: Stage 1 - Geometry Optimization/ relaxation
def make_gpaw_relax(jdata, mdata):
    out_dir = jdata["out_dir"]
    cwd = os.getcwd()
    work_dir = os.path.join(out_dir, global_dirname_02)
    assert os.path.isdir(work_dir)
    work_dir = os.path.abspath(work_dir)

    gpaw_input_name = os.path.basename(jdata["relax_incar"])        # file_name set in the .param file (only name, not path)
    gpaw_runfile_path = os.path.join(work_dir, gpaw_input_name)     # file_path is generated in work_dir
    shutil.copy2(jdata["relax_incar"], gpaw_runfile_path)           # copy the gpaw_input_name to the work_dir, now called "base_file"

    ### Generate symlinks for GPAW input files
    os.chdir(work_dir)
    sys_list = glob.glob("sys-*")
    for ss in sys_list:
        os.chdir(ss)
        ln_src = os.path.relpath(gpaw_runfile_path)     # remmeber the base_file path
        try:
            os.symlink(ln_src, gpaw_input_name)         # create a symlink (has name: gpaw_input_name) to the base_file
        except FileExistsError:
            pass
        os.chdir(work_dir)

    os.chdir(cwd)
    symlink_user_forward_files(
        mdata=mdata,
        task_type="fp",
        work_path=os.path.join(os.path.basename(out_dir), global_dirname_02),
        task_format={"fp": "sys-*"},
    )
    return


def run_gpaw_relax(jdata, mdata):
    check_gpaw_input(jdata["relax_incar"])
    gpaw_input_name = os.path.basename(jdata["relax_incar"])
    fp_command = mdata["fp_command"] + gpaw_input_name
    work_dir = os.path.join(jdata["out_dir"], global_dirname_02)

    forward_files = ["POSCAR", gpaw_input_name]
    user_forward_files = mdata.get("fp" + "_user_forward_files", [])
    forward_files += [os.path.basename(file) for file in user_forward_files]
    backward_files = ["CONF_ASE.traj", "calc.txt", "fp.log"]
    backward_files += mdata.get("fp" + "_user_backward_files", [])
    forward_common_files = []

    relax_tasks = glob.glob(os.path.join(work_dir, "sys-*"))
    relax_tasks.sort()
    # dlog.info("work_dir",work_dir)
    # dlog.info("relax_tasks",relax_tasks)
    if len(relax_tasks) == 0:
        return

    run_tasks = [os.path.basename(ii) for ii in relax_tasks]

    ### Submit the jobs
    check_api_version(mdata)

    submission = make_submission(
        mdata["fp_machine"],
        mdata["fp_resources"],
        commands=[fp_command],
        work_path=work_dir,
        run_tasks=run_tasks,
        group_size=mdata["fp_group_size"],
        forward_common_files=forward_common_files,
        forward_files=forward_files,
        backward_files=backward_files,
        outlog="fp.log",
        errlog="fp.log",
    )
    submission.run_submission()

    ### Convert `CONF_ASE.traj` to `CONTCAR` to be used in the next step
    for ii in relax_tasks:
        if os.path.isfile(f"{ii}/CONF_ASE.traj"):
            traj = Trajectory(f"{ii}/CONF_ASE.traj")
            write_vasp(f"{ii}/CONTCAR", traj[-1])


##### ANCHOR: Stage 2 - scale and perturb
# Use the same `make_scale(jdata)` function as VASP
def pert_scaled_gpaw(jdata):
    ### Extract data from jdata
    out_dir = jdata["out_dir"]
    scale = jdata["scale"]
    pert_box = jdata["pert_box"]
    pert_atom = jdata["pert_atom"]
    pert_numb = jdata["pert_numb"]
    from_poscar = jdata.get("from_poscar", False)

    ### Get the current working directory and the system path
    cwd = os.getcwd()
    path_sp = os.path.join(out_dir, global_dirname_03)
    assert os.path.isdir(path_sp)
    os.chdir(path_sp)
    sys_pe = glob.glob("sys-*")
    sys_pe.sort()
    os.chdir(cwd)

    ### Construct the perturbation command (note: current file is already in the tools directory)
    python_exec = os.path.join(os.path.dirname(__file__), "create_random_disturb.py")
    pert_cmd = sys.executable + f" {python_exec} -etmax {pert_box} -ofmt vasp POSCAR {pert_numb} {pert_atom} > /dev/null"

    ### Loop over each system and scale
    for ii in sys_pe:
        for jj in scale:
            path_work = os.path.join(path_sp, ii, f"scale-{jj:.3f}")
            assert os.path.isdir(path_work)
            os.chdir(path_work)
            sp.check_call(pert_cmd, shell=True)

            ### Loop over each perturbation
            for kk in range(pert_numb):
                pos_in = f"POSCAR{kk}.vasp"
                dir_out = f"{kk:06d}"
                create_path(dir_out)
                pos_out = os.path.join(dir_out, "POSCAR")
                if not from_poscar:
                    poscar_shuffle(pos_in, pos_out)
                else:
                    shutil.copy2(pos_in, pos_out)
                os.remove(pos_in)

            os.chdir(cwd)
    return


##### ANCHOR: Stage 3 - run AIMD
def make_gpaw_md(jdata, mdata):
    out_dir = jdata["out_dir"]
    scale = jdata["scale"]
    pert_numb = jdata["pert_numb"]

    ### Get the current working directory and the system path
    cwd = os.getcwd()
    path_ps = os.path.join(out_dir, global_dirname_03)
    path_ps = os.path.abspath(path_ps)
    assert os.path.isdir(path_ps)
    os.chdir(path_ps)
    sys_ps = glob.glob("sys-*")
    sys_ps.sort()
    os.chdir(cwd)
    path_md = os.path.join(out_dir, global_dirname_04)
    path_md = os.path.abspath(path_md)
    create_path(path_md)

    ### Copy the GPAW input file to the MD path
    gpaw_input_name = os.path.basename(jdata["md_incar"])
    gpaw_runfile_path = os.path.join(path_md, gpaw_input_name)
    shutil.copy2(jdata["md_incar"], gpaw_runfile_path)

    ### Loop over each system, scale, and perturbation number
    for ii in sys_ps:
        for jj in scale:
            for kk in range(pert_numb):
                path_work = os.path.join(path_md, ii, f"scale-{jj:.3f}", f"{kk:06d}")
                create_path(path_work)
                os.chdir(path_work)

                path_pos = os.path.join(path_ps, ii, f"scale-{jj:.3f}", f"{kk:06d}")
                init_pos = os.path.join(path_pos, "POSCAR")
                shutil.copy2(init_pos, "POSCAR")
                try:
                    os.symlink(os.path.relpath(gpaw_runfile_path), gpaw_input_name)
                except FileExistsError:
                    pass
                os.chdir(cwd)

    symlink_user_forward_files(
        mdata=mdata,
        task_type="fp",
        work_path=os.path.join(os.path.basename(out_dir), global_dirname_04),
        task_format={"fp": "sys-*/scale*/00*"},
    )
    return


def run_gpaw_md(jdata, mdata):
    check_gpaw_input(jdata["md_incar"])
    gpaw_input_name = os.path.basename(jdata["md_incar"])
    fp_command = mdata["fp_command"] + gpaw_input_name
    work_dir = os.path.join(jdata["out_dir"], global_dirname_04)

    forward_files = ["POSCAR", gpaw_input_name]
    user_forward_files = mdata.get("fp" + "_user_forward_files", [])
    forward_files += [os.path.basename(file) for file in user_forward_files]
    backward_files = ["CONF_ASE.traj", "calc.txt", "fp.log"]
    backward_files += mdata.get("fp" + "_user_backward_files", [])
    forward_common_files = []

    path_md = os.path.abspath(work_dir)
    assert os.path.isdir(path_md), "md path should exists"

    md_tasks = glob.glob(os.path.join(work_dir, "sys-*/scale*/00*"))
    md_tasks.sort()
    if len(md_tasks) == 0:
        return

    run_tasks = [ii.replace(work_dir + "/", "") for ii in md_tasks]

    ### Submit the jobs
    check_api_version(mdata)

    submission = make_submission(
        mdata["fp_machine"],
        mdata["fp_resources"],
        commands=[fp_command],
        work_path=work_dir,
        run_tasks=run_tasks,
        group_size=mdata["fp_group_size"],
        forward_common_files=forward_common_files,
        forward_files=forward_files,
        backward_files=backward_files,
        outlog="fp.log",
        errlog="fp.log",
    )
    submission.run_submission()


##### ANCHOR: Stage 4 - collect data
def coll_gpaw_md(jdata):
    out_dir = jdata["out_dir"]
    md_nstep = jdata["md_nstep"]
    scale = jdata["scale"]
    pert_numb = jdata["pert_numb"]
    coll_ndata = jdata["coll_ndata"]

    cwd = os.getcwd()
    path_md = os.path.join(out_dir, global_dirname_04)
    path_md = os.path.abspath(path_md)
    assert os.path.isdir(path_md), "md path should exists"
    os.chdir(path_md)
    sys_md = glob.glob("sys-*")
    sys_md.sort()

    for ii in sys_md:
        os.chdir(ii)
        # convert outcars
        valid_outcars = []
        for jj in scale:
            for kk in range(pert_numb):
                path_work = os.path.join(f"scale-{jj:.3f}", f"{kk:06d}")
                outcar = os.path.join(path_work, "OUTCAR")
                # dlog.info("OUTCAR",outcar)
                if os.path.isfile(outcar):
                    # dlog.info("*"*40)
                    with open(outcar) as fin:
                        nforce = fin.read().count("TOTAL-FORCE")
                    # dlog.info("nforce is", nforce)
                    # dlog.info("md_nstep", md_nstep)
                    if nforce == md_nstep:
                        valid_outcars.append(outcar)
                    elif md_nstep == 0 and nforce == 1:
                        valid_outcars.append(outcar)
                    else:
                        dlog.info(
                            f"WARNING : in directory {os.getcwd()} nforce in OUTCAR is not equal to settings in INCAR"
                        )
        arg_cvt = " "
        if len(valid_outcars) == 0:
            raise RuntimeError(
                f"MD dir: {path_md}: find no valid outcar in sys {ii}, "
                "check if your vasp md simulation is correctly done"
            )

        flag = True
        if ("type_map" in jdata) and isinstance(jdata["type_map"], list):
            type_map = jdata["type_map"]
        else:
            type_map = None
        for oo in valid_outcars:
            if flag:
                _sys = dpdata.LabeledSystem(oo, type_map=type_map)
                if len(_sys) > 0:
                    all_sys = _sys
                    flag = False
                else:
                    pass
            else:
                _sys = dpdata.LabeledSystem(oo, type_map=type_map)
                if len(_sys) > 0:
                    all_sys.append(_sys)
        # create deepmd data
        if all_sys.get_nframes() >= coll_ndata:
            all_sys = all_sys.sub_system(np.arange(coll_ndata))
        all_sys.to_deepmd_raw("deepmd")
        all_sys.to_deepmd_npy("deepmd", set_size=all_sys.get_nframes())
        os.chdir(path_md)
    os.chdir(cwd)

    print(" not implemented yet")

    return


##### ANCHOR: Support functions
def check_gpaw_input(input_file: str) -> None:
    """
    Check the input files for the GPAW calculation, to ensure some necessary fields are set.
    """
    with open(input_file) as f:
        text = f.read()

    if "calc.txt" not in text:
        raise ValueError(
            f"The GPAW calculator in file {input_file} did not contain field: txt='calc.txt'. It should be set for backward files."
        )

    if "CONF_ASE.traj" not in text:
        raise ValueError(
            f"The GPAW input file {input_file} did not output the trajectory file 'CONF_ASE.traj'. It should be set for backward files."
        )
    return
