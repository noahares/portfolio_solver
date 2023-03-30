use polars::prelude::IntoLazy;

use portfolio_solver::{
    csv_parser::Data,
    datastructures::*,
    portfolio_simulator::{portfolio_run_from_samples, simulate},
};
use std::path::PathBuf;
mod common;
use common::*;

#[test]
fn test_simple_model_simulation() {
    let config = Config {
        files: vec![
            PathBuf::from("data/test/algo1.csv"),
            "data/test/algo2.csv".into(),
        ],
        ..default_config()
    };
    CONFIG.set(config).ok();
    DF_CONFIG.set(DataframeConfig::new()).ok();
    let data = Data::new().unwrap();
    let portfolio = Portfolio {
        name: "final_portfolio_opt".to_string(),
        resource_assignments: vec![
            (
                Algorithm {
                    algorithm: "algo1".into(),
                    num_threads: 1,
                },
                0.0,
            ),
            (
                Algorithm {
                    algorithm: "algo2".into(),
                    num_threads: 1,
                },
                2.0,
            ),
        ],
    };
    let simulation_df = simulate(&data.df, &portfolio, 42)
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(simulation_df.height(), 8);
    assert!(!simulation_df
        .column("algorithm")
        .unwrap()
        .utf8()
        .unwrap()
        .into_no_null_iter()
        .any(|s| s == "algo1"));
    let portfolio_df = portfolio_run_from_samples(
        simulation_df.lazy(),
        &["instance", "k", "feasibility_threshold"],
        &["algorithm", "num_threads"],
        2,
        "portfolio",
    )
    .collect()
    .unwrap();
    assert_eq!(portfolio_df.height(), 4);
    assert_eq!(
        portfolio_df
            .sort(["quality"], false)
            .unwrap()
            .column("quality")
            .unwrap()
            .f64()
            .unwrap()
            .to_ndarray()
            .unwrap(),
        ndarray::Array1::from_vec(vec![9.0, 11.0, 18.0, 24.0])
    );
}
