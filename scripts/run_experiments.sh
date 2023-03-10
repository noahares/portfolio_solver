#!/bin/sh

for portfolio in 1 2 3 4; do
  for slowdown_ratio in 1.0 0.5 0.25 0.2 0.1 0.05; do
    ./target/release/portfolio_solver -c config/portfolio${portfolio}.json -s $slowdown_ratio -o results/portfolio${portfolio}_${slowdown_ratio} -t 600
    ./target/release/portfolio_executor results/portfolio${portfolio}_${slowdown_ratio}/executor.json
    python scripts/performance_profiles.py results/portfolio${portfolio}_${slowdown_ratio}/portfolio km1 results/portfolio${portfolio}_${slowdown_ratio}/execution.csv
    python scripts/time_plot.py results/portfolio${portfolio}_${slowdown_ratio}/portfolio totalPartitonTime results/portfolio${portfolio}_${slowdown_ratio}/execution.csv
  done
done
