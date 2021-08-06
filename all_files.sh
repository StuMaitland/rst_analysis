#!/bin/bash
# NOTE : Quote it else use array to avoid problems #
FILES="/Users/stuartbman/GitHub/rst_analysis/data/*"
for f in $FILES
do
  echo "Processing $f file..."
  python3 force_corr.py -i "$f"
  # take action on each file. $f store current file name
done