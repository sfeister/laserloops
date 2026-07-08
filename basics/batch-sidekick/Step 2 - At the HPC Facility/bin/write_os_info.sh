#!/usr/bin/env bash
#
# Written by ChatGPT with help from Scott Feister on 2026-07-06.
#
# Write basic operating system information to a text file.
#
# Usage:
#
#   write_os_info.sh os_info.txt

set -euo pipefail

output_file="$1"

{
    echo "Date:"
    date
    echo

    echo "Hostname:"
    hostname
    echo

    echo "Kernel:"
    uname -a
    echo

    echo "OS release:"
    cat /etc/os-release
} > "$output_file"