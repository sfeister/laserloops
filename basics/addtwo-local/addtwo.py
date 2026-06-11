"""
Read two numbers from input files, add them, and write the result.

Inputs:
    inputs/a.txt
    inputs/b.txt

Output:
    outputs/c.txt

Created by ChatGPT.
"""

from pathlib import Path


INPUT_A = Path("inputs/a.txt")
INPUT_B = Path("inputs/b.txt")
OUTPUT_C = Path("outputs/c.txt")


def read_number(filename):
    """Read one number from a text file."""
    text = filename.read_text().strip()
    return float(text)


def write_number(filename, number):
    """Write one number to a text file."""
    filename.parent.mkdir(exist_ok=True)
    filename.write_text(f"{number:g}\n")


try:
    a = read_number(INPUT_A)
    b = read_number(INPUT_B)

    c = a + b

    write_number(OUTPUT_C, c)

except Exception as error:
    OUTPUT_C.parent.mkdir(exist_ok=True)
    OUTPUT_C.write_text(f"ERROR\n{error}\n")