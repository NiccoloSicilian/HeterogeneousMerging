#!/bin/bash
REPO_DIR="/leonardo_scratch/fast/IscrC_eff-SAM2/HeterogeneousMerging/HeterogeneousMerging"
cd "$REPO_DIR" || { echo "Directory not found: $REPO_DIR"; exit 1; }
git pull
echo "Done."
