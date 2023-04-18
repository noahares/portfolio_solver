use portfolio_solver::{csv_parser::Data, datastructures::*};
mod common;
use common::*;
use ndarray::arr1;
use std::path::PathBuf;

#[test]
fn test_slowdown_ratio_filter() {
    let config = Config {
        files: vec![
            PathBuf::from("data/test/algo1.csv"),
            "data/test/algo6.csv".into(),
        ],
        slowdown_ratio: 2.0,
        ..default_config()
    };
    CONFIG.set(config).ok();
    let data = Data::new().unwrap();
    assert_eq!(data.num_instances, 4);
    assert_eq!(data.num_algorithms, 2);
    assert_eq!(data.best_per_instance_time, arr1(&[1.2, 4.2, 2.0, 3.0]));
}
