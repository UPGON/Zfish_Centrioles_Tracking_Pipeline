#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:11:05 2024

@author: floriancurvaia
"""
import os
from pathlib import Path
import argparse

SLURM_COMMAND = """#!/usr/bin/bash

#SBATCH --account=upgon
#SBATCH --array=0-{0}%100
#SBATCH --qos=serial
#SBATCH --mem=36000
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH -o /home/curvaia/outputs/nuc_seg_time/out_%a.txt
#SBATCH -e /home/curvaia/outputs/nuc_seg_time/errors_%a.txt

exec python /home/curvaia/time_track_nuc_coords_parallel.py $SLURM_ARRAY_TASK_ID {1}
"""


def main():
    CLI = argparse.ArgumentParser()
    CLI.add_argument('in_fld', type=str)
    args = CLI.parse_args()
    path_in=Path(args.in_fld)
    print(path_in)
    all_tp=set([int(fn.name.split("T")[1].split("_")[0]) for fn in list(path_in.glob("*.npy"))])
    n=max(all_tp)

    command = SLURM_COMMAND.format(n , args.in_fld)
    print(command)
    with open("temp.sh", "w") as f:
        f.write(command)
    os.system("sbatch temp.sh")
    os.unlink("temp.sh")


if __name__ == "__main__":
    main()
