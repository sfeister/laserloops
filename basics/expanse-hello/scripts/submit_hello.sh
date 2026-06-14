#!/usr/bin/env bash
# Written by ChatGPT on June 14, 2026, with guidance from Scott Feister.
#
# Create a fresh scratch run directory, copy this repository into it as a
# source snapshot, and submit the hello-world Slurm job on Expanse.
#
# Assumptions:
#   - This script is run from anywhere inside the canonical repo, or by path.
#   - The canonical repo contains:
#       src/hello.py
#       jobs/hello.sbatch
#   - The Slurm job expects the copied source snapshot to appear at:
#       source/src/hello.py
#   - The Expanse account/allocation is provided through EXPANSE_ACCOUNT.
#   - The run will be created under:
#       /expanse/lustre/scratch/$USER/temp_project/expanse-hello/
#
# Usage:
#   EXPANSE_ACCOUNT=YOUR_PROJECT scripts/submit_hello.sh
#
# Optional:
#   RUN_NAME=hello-001 EXPANSE_ACCOUNT=YOUR_PROJECT scripts/submit_hello.sh

set -euo pipefail

PROJECT_NAME="expanse-hello"

if [[ -z "${EXPANSE_ACCOUNT:-}" ]]; then
    echo "ERROR: EXPANSE_ACCOUNT is not set."
    echo
    echo "Example:"
    echo "  EXPANSE_ACCOUNT=YOUR_PROJECT scripts/submit_hello.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_NAME="${RUN_NAME:-hello-$TIMESTAMP}"

LUSTRE_SCRATCH="/expanse/lustre/scratch/$USER/temp_project" # true all across Expanse
RUN_DIR="$LUSTRE_SCRATCH/$PROJECT_NAME/$RUN_NAME"

echo "Repository:  $REPO_DIR"
echo "Run name:    $RUN_NAME"
echo "Run dir:     $RUN_DIR"
echo "Account:     $EXPANSE_ACCOUNT"
echo

mkdir -p "$RUN_DIR"

echo "Copying source snapshot..."
rsync -av \
    --exclude ".git" \
    --exclude "__pycache__" \
    --exclude ".venv" \
    --exclude "logs" \
    --exclude "outputs" \
    --exclude "tmp" \
    "$REPO_DIR/" \
    "$RUN_DIR/source/"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/outputs" "$RUN_DIR/tmp"

cd "$RUN_DIR"

echo
echo "Submitting from:"
pwd
echo

sbatch -A "$EXPANSE_ACCOUNT" source/jobs/hello.sbatch