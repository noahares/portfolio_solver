use portfolio_solver::{csv_parser::Data, datastructures::*};
mod common;
use common::*;
use ndarray::arr1;
use std::path::PathBuf;

#[test]
fn test_invalid_rows() {
    let config = Config {
        files: vec![PathBuf::from("data/test/algo4.csv")],
        ..default_config()
    };
    CONFIG.set(config).ok();
    DF_CONFIG.set(DataframeConfig::new()).ok();
    let data = Data::new().unwrap();
    assert_eq!(data.num_instances, 4);
    assert_eq!(data.num_algorithms, 1);
    assert_eq!(data.best_per_instance, arr1(&[20.0, 20.0, 20.0, 20.0]));
}
