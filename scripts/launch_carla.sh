#!/bin/bash

# Launch CARLA server
./CarlaUE4.sh -RenderOffScreen -quality-level=Epic -fps=20 -world-port=2000 &

# Wait for server to initialize
sleep 10

# Run Python script
python train.py --config configs/urban.yaml

# Cleanup
pkill -f CarlaUE4