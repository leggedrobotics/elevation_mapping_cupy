#!/bin/bash
cd ~/workspace
vcs import < src/elevation_mapping_cupy/docker/src.repos src/ --recursive -w $(($(nproc)/2))

sudo apt update
rosdep update
rosdep install --from-paths src --ignore-src -y -r