use portfolio_solver::{csv_parser::Data, datastructures::*, solver::solve};
mod common;
use common::*;
use std::path::PathBuf;

#[test]
fn test_seq_vs_par() {
    let config = Config {
        files: vec![
            PathBuf::from("data/test/algo1.csv"),
            "data/test/algo7.csv".into(),
        ],
        num_cores: 8,
        ..default_config()
    };
    let k = config.num_cores;
    CONFIG.set(config).ok();
    DF_CONFIG.set(DataframeConfig::new()).ok();
    let data = Data::new().unwrap();
    assert_eq!(
        solve(&data, k as usize, Timeout::default())
            .unwrap()
            .final_portfolio,
        Portfolio {
            name: "final_portfolio_opt".to_string(),
            resource_assignments: vec![
                (
                    Algorithm {
                        algorithm: "algo1".into(),
                        num_threads: 1
                    },
                    4.0
                ),
                (
                    Algorithm {
                        algorithm: "algo7".into(),
                        num_threads: 4
                    },
                    1.0
                ),
            ]
        }
    );
}
