#!/usr/bin/env bash
# Written by ChatGPT on June 14, 2026, with guidance from Scott Feister.
#
# Create a fresh scratch run directory, copy this repository into it as a
# source snapshot, copy input files into the run directory, and submit the
# addtwo-expanse Slurm job on Expanse.
#
# Assumptions:
#   - This script is run from anywhere inside the canonical repo, or by path.
#   - The canonical repo contains:
#       src/addtwo.py
#       jobs/addtwo.sbatch
#       inputs/a.txt
#       inputs/b.txt
#   - The Slurm job expects the copied source snapshot to appear at:
#       source/src/addtwo.py
#   - addtwo.py expects run-specific inputs and outputs at:
#       inputs/a.txt
#       inputs/b.txt
#       outputs/c.txt
#   - For now, run-specific inputs are copied from:
#       source/inputs/
#   - The Expanse account/allocation is provided through EXPANSE_ACCOUNT.
#   - The run will be created under:
#       /expanse/lustre/scratch/$USER/temp_project/addtwo-expanse/
#
# Usage:
#   EXPANSE_ACCOUNT=YOUR_PROJECT scripts/submit_addtwo.sh
#
# Optional:
#   RUN_NAME=addtwo-001 EXPANSE_ACCOUNT=YOUR_PROJECT scripts/submit_addtwo.sh

set -euo pipefail

PROJECT_NAME="addtwo-expanse"
SCRATCH_BASE="/expanse/lustre/scratch/$USER/temp_project"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_NAME="${RUN_NAME:-addtwo-$TIMESTAMP}"
RUN_DIR="$SCRATCH_BASE/$PROJECT_NAME/$RUN_NAME"


require_env_var() {
    local name="$1"

    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: Required environment variable is not set: $name" >&2
        echo >&2
        echo "Example:" >&2
        echo "  $name=YOUR_PROJECT scripts/submit_addtwo.sh" >&2
        exit 1
    fi
}


require_file() {
    local path="$1"

    if [[ ! -f "$path" ]]; then
        echo "ERROR: Expected file not found: $path" >&2
        exit 1
    fi
}


print_summary() {
    echo "Repository:  $REPO_DIR"
    echo "Run name:    $RUN_NAME"
    echo "Run dir:     $RUN_DIR"
    echo "Account:     $EXPANSE_ACCOUNT"
    echo
}


copy_source_snapshot() {
    echo "Copying source snapshot..."

    mkdir -p "$RUN_DIR/source"

    rsync -av \
        --exclude ".git" \
        --exclude "__pycache__" \
        --exclude ".venv" \
        --exclude "logs" \
        --exclude "outputs" \
        --exclude "tmp" \
        "$REPO_DIR/" \
        "$RUN_DIR/source/"
}


prepare_run_directories() {
    echo
    echo "Creating run directories..."

    mkdir -p \
        "$RUN_DIR/inputs" \
        "$RUN_DIR/tmp"
}


copy_run_inputs() {
    echo
    echo "Copying run inputs from source/inputs/ to inputs/..."

    rsync -av \
        "$RUN_DIR/source/inputs/" \
        "$RUN_DIR/inputs/"
}


submit_job() {
    cd "$RUN_DIR"

    echo
    echo "Submitting from:"
    pwd
    echo

    sbatch -A "$EXPANSE_ACCOUNT" source/jobs/addtwo.sbatch
}


main() {
    require_env_var "EXPANSE_ACCOUNT"

    require_file "$REPO_DIR/src/addtwo.py"
    require_file "$REPO_DIR/jobs/addtwo.sbatch"
    require_file "$REPO_DIR/inputs/a.txt"
    require_file "$REPO_DIR/inputs/b.txt"

    print_summary
    copy_source_snapshot
    prepare_run_directories
    copy_run_inputs
    submit_job
}


main "$@"