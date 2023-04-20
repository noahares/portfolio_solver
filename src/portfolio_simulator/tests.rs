use polars::prelude::*;

use crate::{
    datastructures::*,
    portfolio_simulator::{portfolio_run_from_samples, simulate},
};

#[test]
fn test_simple_model_simulation() {
    let df = df! {
        "algorithm" => ["algo1", "algo1", "algo2", "algo2", "algo3", "algo3"],
        "num_threads" => vec![1; 6],
        "instance" => ["graph1", "graph2", "graph1", "graph2", "graph1", "graph2"],
        "quality" => [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "time" => [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "valid" => vec![true; 6],
    }.unwrap();
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
    let simulation_df =
        simulate(&df, &portfolio, 42).unwrap().collect().unwrap();
    assert_eq!(simulation_df.height(), 4);
    assert!(!simulation_df
        .column("algorithm")
        .unwrap()
        .utf8()
        .unwrap()
        .into_no_null_iter()
        .any(|s| s == "algo1"));
}

#[test]
fn test_simple_model_simulation_from_samples() {
    let df = df! {
        "algorithm" => ["algo2", "algo2", "algo2", "algo2"],
        "num_threads" => vec![1; 4],
        "instance" => ["graph1", "graph2", "graph1", "graph2"],
        "quality" => [1.0, 2.0, 3.0, 4.0],
        "time" => [1.0, 1.0, 1.0, 1.0],
        "valid" => vec![true; 4],
    }
    .unwrap();
    let portfolio_df = portfolio_run_from_samples(
        df.lazy(),
        &["instance"],
        &["algorithm", "num_threads"],
        4,
        "portfolio",
    )
    .collect()
    .unwrap();
    assert_eq!(portfolio_df.height(), 2);
    dbg!(&portfolio_df);
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
        ndarray::Array1::from_vec(vec![1.0, 2.0])
    );
}
