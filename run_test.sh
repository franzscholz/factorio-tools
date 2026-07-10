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

# Everything is ok
exit 0
