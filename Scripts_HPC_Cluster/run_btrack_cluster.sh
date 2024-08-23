#!/usr/bin/env bash

#SBATCH --account=upgon
#SBATCH --qos=serial
#SBATCH --mem=78000
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH -o /home/curvaia/outputs/out_btrack.txt
#SBATCH -e /home/curvaia/outputs/errors_btrack.txt

exec python /home/curvaia/btrack_cluster.py

