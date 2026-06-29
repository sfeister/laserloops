# tiled_helpers.py
# Basic helper functions for connecting to Tiled database

import os
from tiled.client import from_uri

def get_tiled_client():
    """Connect to Tiled using environment variables.
    Simple helper function written by ChatGPT"""

    tiled_uri = os.environ.get("TILED_URI")
    tiled_api_key = os.environ.get("TILED_API_KEY")

    if tiled_uri is None:
        raise RuntimeError(
            "Missing TILED_URI.\n"
            "Example:\n"
            "    export TILED_URI='http://your-tiled-server.lan:8000'\n"
        )

    if tiled_api_key is None:
        raise RuntimeError(
            "Missing TILED_API_KEY.\n"
            "Example:\n"
            "    export TILED_API_KEY='your-api-key-here'\n"
        )

    return from_uri(tiled_uri, api_key=tiled_api_key)