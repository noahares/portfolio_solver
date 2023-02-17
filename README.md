# Setup

Set up the git-hook for formatting:
```sh
# inside the repo root
ln -s ../../scripts/pre-commit .git/hooks/pre-commit
```

# Run

```sh
# to generate lower bounds for graph instances
cargo run --release --bin quality_lb config/m_h_quality_lb.json
# to generate lower bounds for hypergraph instances
cargo run --release --bin quality_lb config/m_hg_quality_lb.json

# to generate a portfolio
cargo run --release --bin portfolio_solver config/<portfolio>.json
```
