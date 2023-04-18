use portfolio_solver::{csv_parser::Data, datastructures::*};
mod common;
use common::*;
use ndarray::arr1;
use std::path::PathBuf;

#[test]
fn test_missing_algo_for_instance() {
    let config = Config {
        files: vec![
            PathBuf::from("data/test/algo2.csv"),
            "data/test/algo5.csv".into(),
        ],
        ..default_config()
    };
    CONFIG.set(config).ok();
    let data = Data::new().unwrap();
    assert_eq!(data.num_instances, 4);
    assert_eq!(data.num_algorithms, 2);
    assert_eq!(data.best_per_instance, arr1(&[16.0, 7.0, 22.0, 9.0]));
}
