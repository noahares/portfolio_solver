use ndarray::{arr1, aview2, Axis};
use polars::prelude::*;

use crate::{
    csv_parser::{utils::filter_algorithms_by_slowdown, Data},
    datastructures::Config,
};

use super::utils::{best_per_instance_count, stats_by_sampling};

use crate::test_utils::*;

#[test]
fn test_dataframe() {
    let config = Config {
        files: vec![
            "data/test/algo1.csv".to_string(),
            "data/test/algo2.csv".into(),
        ],
        ..default_config()
    };
    let data = Data::new(&config);
    assert_eq!(data.num_instances, 4);
    assert_eq!(data.num_algorithms, 2);
    assert_eq!(data.best_per_instance, arr1(&[16.0, 7.0, 18.0, 9.0]));
    assert_eq!(
        data.stats.index_axis(Axis(2), 0),
        aview2(&[[18.0, 16.0], [9.0, 7.0], [18.0, 22.0], [9.0, 9.0]])
    );
}

#[test]
fn test_handle_quality_is_zero() {
    let config = Config {
        files: vec![
            "data/test/algo2.csv".to_string(),
            "data/test/algo3.csv".into(),
        ],
        ..default_config()
    };
    let data = Data::new(&config);
    assert_eq!(data.num_instances, 4);
    assert_eq!(data.num_algorithms, 2);
    assert_eq!(data.best_per_instance, arr1(&[1.0, 7.0, 22.0, 1.0]));
}

#[test]
fn test_handle_invalid_rows() {
    let config = Config {
        files: vec!["data/test/algo4.csv".to_string()],
        ..default_config()
    };
    let data = Data::new(&config);
    assert_eq!(data.num_instances, 4);
    assert_eq!(data.num_algorithms, 1);
    assert_eq!(data.best_per_instance, arr1(&[20.0, 20.0, 20.0, 20.0]));
}

#[test]
fn test_missing_algo_for_instance() {
    let config = Config {
        files: vec![
            "data/test/algo2.csv".to_string(),
            "data/test/algo5.csv".into(),
        ],
        ..default_config()
    };
    let data = Data::new(&config);
    assert_eq!(data.num_instances, 4);
    assert_eq!(data.num_algorithms, 2);
    assert_eq!(data.best_per_instance, arr1(&[16.0, 7.0, 22.0, 9.0]));
}

#[test]
fn test_best_per_instance_time() {
    let config = Config {
        files: vec![
            "data/test/algo1.csv".to_string(),
            "data/test/algo6.csv".into(),
        ],
        ..default_config()
    };
    let data = Data::new(&config);
    assert_eq!(data.num_instances, 4);
    assert_eq!(data.num_algorithms, 2);
    assert_eq!(data.best_per_instance_time, arr1(&[1.2, 4.2, 2.0, 3.0]));
}

#[test]
fn test_slowdown_ratio_filter() {
    let config = Config {
        files: vec![
            "data/test/algo1.csv".to_string(),
            "data/test/algo6.csv".into(),
        ],
        slowdown_ratio: 2.0,
        ..default_config()
    };
    let data = Data::new(&config);
    assert_eq!(data.num_instances, 4);
    assert_eq!(data.num_algorithms, 2);
    assert_eq!(data.best_per_instance_time, arr1(&[1.2, 4.2, 2.0, 3.0]));
}

#[test]
fn test_best_per_instance_count() {
    let instance_fields = &["instance", "k"];
    let algorithm_fields = &["algorithm", "num_threads"];
    let df = df! {
            "instance" => ["graph1", "graph1", "graph1", "graph2", "graph2", "graph2"],
            "k" => vec![2; 6],
            "algorithm" => ["algo1", "algo2", "algo3", "algo1", "algo2", "algo3"],
            "num_threads" => vec![1; 6],
            "quality" => [1.0, 2.0, 2.0, 2.0, 1.0, 2.0],
        }.unwrap();
    let ranking = best_per_instance_count(
        df,
        instance_fields,
        algorithm_fields,
        "quality",
    );
    assert_eq!(
        ranking["count"],
        Series::from_vec("count", vec![1.0, 1.0, 0.0])
    );
}

#[test]
fn test_stats_by_sampling() {
    let instance_fields = &["instance", "k", "feasibility_threshold"];
    let algorithm_fields = &["algorithm", "num_threads"];
    let df = df! {
            "instance" => ["graph1", "graph1", "graph1", "graph1", "graph2", "graph2", "graph2", "graph2"],
            "k" => vec![2; 8],
            "feasibility_threshold" => vec![0.0; 8],
            "algorithm" => ["algo1", "algo1", "algo1", "algo1", "algo1", "algo1", "algo1", "algo1"],
            "num_threads" => vec![1; 8],
            "quality" => [10.0, 8.0, 9.0, 7.0, 20.0, 18.0, 22.0, 19.0],
        }.unwrap();
    let stats_df =
        stats_by_sampling(df.lazy(), 4, instance_fields, algorithm_fields)
            .collect()
            .unwrap();
    dbg!(&stats_df["e_min"]);
    assert_eq!(
        stats_df["e_min"],
        Series::from_vec(
            "e_min",
            vec![9.0, 7.0, 7.0, 7.0, 22.0, 19.0, 18.0, 18.0]
        )
    );
}

#[test]
fn test_algorithm_slowdown_filtering() {
    let instance_fields = &["instance"];
    let algorithm_fields = &["algorithm"];
    let df = df! {
            "instance" => ["graph1", "graph1", "graph1", "graph2", "graph2", "graph2"],
            "algorithm" => ["algo1", "algo2", "algo3", "algo1", "algo2", "algo3"],
            "quality" => [1.0, 2.0, 2.0, 2.0, 1.0, 2.0],
            "time" => [2.0, 2.0, 1.0, 2.0, 2.0, 1.0],
        }.unwrap();
    let filtered_df = filter_algorithms_by_slowdown(
        df.lazy(),
        instance_fields,
        algorithm_fields,
        0.5,
    )
    .collect()
    .unwrap();
    assert_eq!(
        filtered_df["algorithm"],
        Series::new("algorithm", &["algo3".to_string(), "algo3".into()])
    );
}
