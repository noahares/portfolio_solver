#!/bin/sh

run () {
  echo "Running portfolio $1 with slowdown ratio $2 and k = $3"
  portfolio=$1
  slowdown_ratio=$2
  k=$3
  ./target/release/portfolio_solver -c config/portfolio${portfolio}.json -s $slowdown_ratio -o results/portfolio${portfolio}_${slowdown_ratio}_${k} -k ${k} -i -r -t 600
  exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    ./target/release/portfolio_executor -c results/portfolio${portfolio}_${slowdown_ratio}_${k}/executor.json
    python scripts/performance_profiles.py results/portfolio${portfolio}_${slowdown_ratio}_${k}/portfolio km1 results/portfolio${portfolio}_${slowdown_ratio}_${k}/execution.csv &> /dev/null
    python scripts/time_plot.py results/portfolio${portfolio}_${slowdown_ratio}_${k}/portfolio totalPartitionTime results/portfolio${portfolio}_${slowdown_ratio}_${k}/execution.csv &> /dev/null
  else
    echo "$1 $2 $3 failed with exit code $exit_code" >> portfolio_solver.log
  fi
}

for portfolio in 1 2 3 4 5; do
  for slowdown_ratio in 2.0 1.0 0.5 0.25 0.1; do
    if [[ $portfolio == 5 ]]; then
      run $portfolio $slowdown_ratio 256
    else
      for k in 5 10 32; do
        run $portfolio $slowdown_ratio $k
      done
    fi
  done
done
