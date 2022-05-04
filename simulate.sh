#! /bin/bash

echo "Preparing data"
python3 prep.py
echo
echo "Normalising..."
python3 normalise.py
echo
echo "Generating matrices"
# python3 pulse_invert.py
echo
echo "Solving"
python3 pulse_prop.py
