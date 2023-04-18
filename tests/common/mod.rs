use std::path::PathBuf;

use portfolio_solver::datastructures::{Config, Timeout};

pub fn default_config() -> Config {
    Config {
        files: vec![],
        graphs: PathBuf::new(),
        ks: vec![2, 4],
        feasibility_thresholds: vec![0.03],
        num_cores: 2,
        slowdown_ratio: std::f64::MAX,
        num_seeds: 1,
        out_dir: PathBuf::new(),
        timeout: Timeout::default(),
    }
}