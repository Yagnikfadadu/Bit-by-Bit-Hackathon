#!/bin/bash

OUTDIR=~/traces
mkdir -p $OUTDIR

EVENTS="cycles,instructions,branches,branch-misses,cache-references,cache-misses,LLC-loads,LLC-load-misses"

for i in {1..5}; do
  echo "Collecting trace run $i..."
  perf stat -I 50 -e $EVENTS -- /home/hackathon/dist/model_inference \
    > $OUTDIR/trace_run_$i.txt
done

echo "All traces saved in $OUTDIR"
