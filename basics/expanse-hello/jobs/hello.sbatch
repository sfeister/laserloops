#!/bin/bash
# Written by ChatGPT on June 14, 2026, with guidance from Scott Feister.
#
# Minimal Slurm batch script for running a Python "hello world" job on Expanse.
#
# Assumptions:
#   - This script is submitted with sbatch from a per-run scratch directory.
#   - The run directory contains a copied source snapshot at:
#       source/src/hello.py
#   - The run directory is allowed to create/use:
#       logs/
#       outputs/
#       tmp/
#   - The Expanse account/allocation is passed at submit time, for example:
#       sbatch -A YOUR_PROJECT source/jobs/hello.sbatch
#   - The job is small enough for the debug partition.
#   - Python is provided through the Expanse module environment.
#
# Expected run directory layout:
#
#   run-001/
#   ├── source/
#   │   ├── src/
#   │   │   └── hello.py
#   │   └── jobs/
#   │       └── hello.sbatch
#   ├── logs/
#   ├── outputs/
#   └── tmp/
#
# This script writes Slurm stdout/stderr to logs/ and expects hello.py to write
# its program output to outputs/.
#
#SBATCH --job-name=hello-python
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:00:30
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --export=ALL

module purge
module load cpu
module load slurm
module load python

mkdir -p logs

python source/src/hello.py
