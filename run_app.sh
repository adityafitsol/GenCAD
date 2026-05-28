#!/bin/bash
# Script to run the GenCAD Flask GUI
# This uses xvfb-run to allow headless 3D geometry generation

# Default port
PORT=${1:-5000}

echo "Starting GenCAD Flask GUI on port $PORT..."
echo "Note: Make sure you have downloaded the checkpoints to model/ckpt/"

# Check if xvfb-run is installed
if command -v xvfb-run &> /dev/null; then
    xvfb-run --server-args="-screen 0 2048x2048x24" python app.py
else
    echo "Warning: xvfb-run not found. If generation fails, please install it (sudo apt install xvfb)."
    python app.py
fi
