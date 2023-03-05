#!/bin/sh

for portfolio in 1 2 3 4; do
  for slowdown_ratio in 2.0 3.0 4.0 5.0 10.0 20.0; do
    ./target/release/portfolio_solver -c config/portfolio${portfolio}.json -s $slowdown_ratio -o results/portfolio${portfolio}_${slowdown_ratio} -t 900
    ./target/release/portfolio_executor results/portfolio${portfolio}_${slowdown_ratio}/executor.json
  done
done
