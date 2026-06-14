#!/bin/bash
# Written by ChatGPT on June 14, 2026, with guidance from Scott Feister.
#
# Minimal Slurm batch script for running addtwo.py on Expanse.
#
# Assumptions:
#   - This script is submitted with sbatch from a per-run scratch directory.
#   - The run directory contains a copied source snapshot at:
#       source/src/addtwo.py
#   - The run directory contains input files at:
#       inputs/a.txt
#       inputs/b.txt
#   - The run directory is allowed to create/use:
#       logs/
#       outputs/
#       tmp/
#   - The Expanse account/allocation is passed at submit time, for example:
#       sbatch -A YOUR_PROJECT source/jobs/addtwo.sbatch
#   - The job is small enough for the debug partition.
#   - Python is provided through the Expanse module environment.
#
# Expected run directory layout:
#
#   run-001/
#   ├── source/
#   │   ├── src/
#   │   │   └── addtwo.py
#   │   └── jobs/
#   │       └── addtwo.sbatch
#   ├── inputs/
#   │   ├── a.txt
#   │   └── b.txt
#   ├── logs/
#   ├── outputs/
#   └── tmp/
#
# This script writes Slurm stdout/stderr to logs/.
# addtwo.py reads inputs/a.txt and inputs/b.txt, then writes outputs/c.txt.
#
#SBATCH --job-name=addtwo
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:00:30
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --export=ALL

set -euo pipefail

module purge
module load cpu
module load slurm

mkdir -p logs

echo "Starting addtwo job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job account: ${SLURM_JOB_ACCOUNT:-not set}"
echo "Node: ${SLURMD_NODENAME}"
echo "Run directory: $(pwd)"
echo

python source/src/addtwo.py .

echo
echo "Finished addtwo job"