use ndarray::{arr1, aview2, Axis};
use portfolio_solver::{csv_parser::Data, datastructures::*};
mod common;
use common::*;
use std::path::PathBuf;

#[test]
fn test_dataframe() {
    let config = Config {
        files: vec![
            PathBuf::from("data/test/algo1.csv"),
            "data/test/algo2.csv".into(),
        ],
        ..default_config()
    };
    CONFIG.set(config).ok();
    let data = Data::new().unwrap();
    assert_eq!(data.num_instances, 4);
    assert_eq!(data.num_algorithms, 2);
    assert_eq!(data.best_per_instance, arr1(&[16.0, 7.0, 18.0, 9.0]));
    assert_eq!(
        data.stats.index_axis(Axis(2), 0),
        aview2(&[[18.0, 16.0], [9.0, 7.0], [18.0, 22.0], [9.0, 9.0]])
    );
}
