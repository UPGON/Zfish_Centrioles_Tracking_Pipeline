#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:47:13 2024

@author: floriancurvaia
"""

import os
from pathlib import Path
import argparse

SLURM_COMMAND = """#!/usr/bin/bash

#SBATCH --account=upgon
#SBATCH --array=0-{0}%100
#SBATCH --qos=serial
#SBATCH --mem=22000
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH -o /home/curvaia/outputs/stardist_live/out_%a.txt
#SBATCH -e /home/curvaia/outputs/stardist_live/errors_%a.txt

exec python /home/curvaia/stardist_nuc_seg_live.py $SLURM_ARRAY_TASK_ID {1}
"""


def main():
    CLI = argparse.ArgumentParser()
    CLI.add_argument('in_fld', type=str)
    args = CLI.parse_args()
    path_in=Path(args.in_fld)
    print(path_in)
    all_tp=set([int(fn.name.split("T")[1].split(".")[0]) for fn in list(path_in.glob("*C01*.tif"))])
    n=max(all_tp)

    command = SLURM_COMMAND.format(n , args.in_fld)
    print(command)
    with open("temp.sh", "w") as f:
        f.write(command)
    os.system("sbatch temp.sh")
    os.unlink("temp.sh")


if __name__ == "__main__":
    main()
