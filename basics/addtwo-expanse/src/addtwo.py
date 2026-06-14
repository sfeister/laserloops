#!/usr/bin/env python3
"""
Add two numbers from a run directory.

Expected run directory layout:

    RUN_DIR/
    ├── inputs/
    │   ├── a.txt
    │   └── b.txt
    └── outputs/
        └── c.txt

The inputs/ directory and input files must already exist.
The outputs/ directory is created automatically if needed.
No output file is written if the calculation fails.

Usage:

    python source/src/addtwo.py RUN_DIR

Example from inside a run directory:

    python source/src/addtwo.py .

Created by ChatGPT with guidance from Scott Feister on June 14, 2026.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys


def read_number(path: Path) -> float:
    """Read one floating-point number from a text file."""
    text = path.read_text().strip()
    return float(text)


def write_number(path: Path, number: float) -> None:
    """Write one number to a text file, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{number:g}\n")


def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = ArgumentParser(
        description="Read inputs/a.txt and inputs/b.txt from RUN_DIR, then write outputs/c.txt."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to the run directory containing inputs/.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the calculation."""
    args = parse_args()

    run_dir = args.run_dir.resolve()

    input_a = run_dir / "inputs" / "a.txt"
    input_b = run_dir / "inputs" / "b.txt"
    output_c = run_dir / "outputs" / "c.txt"

    try:
        a = read_number(input_a)
        b = read_number(input_b)
        c = a + b
        write_number(output_c, c)

    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())