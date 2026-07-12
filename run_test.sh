#! /usr/bin/env bash
#
# Run tests.
#
# SPDX-License-Identifier: MIT

set -eou pipefail

# Activate venv
source .venv/bin/activate

# Run Jupyter notebook
jupyter-execute py_neutron_capture.ipynb

# Run the simulation
python py_neutron_capture_sim.py

# Everything is ok
exit 0
