"""TetherCache — minimal setup script.

Usage::

    python setup.py develop

This makes the ``tethercache``, ``wan``, ``pipeline``, and ``utils``
top-level packages importable from the repo root, matching the layout
the bundled inference scripts expect.
"""
from setuptools import find_packages, setup

setup(
    name="tethercache",
    version="0.1.0",
    description=(
        "TetherCache: Stabilizing Long-Form Video Generation with Gated "
        "Recall and Trusted Alignment."
    ),
    packages=find_packages(
        include=[
            "tethercache", "tethercache.*",
            "wan", "wan.*",
            "pipeline", "pipeline.*",
            "utils", "utils.*",
        ]
    ),
    python_requires=">=3.10",
)
