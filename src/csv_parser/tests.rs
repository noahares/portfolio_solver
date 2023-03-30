use super::utils::{best_per_instance_count, stats_by_sampling};
use crate::csv_parser::utils::filter_algorithms_by_slowdown;
use crate::datastructures::{DataframeConfig, DF_CONFIG};
use polars::prelude::*;

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
    )
    .unwrap();
    assert_eq!(
        ranking["count"],
        Series::from_vec("count", vec![1.0, 1.0, 0.0])
    );
}

#[test]
fn test_stats_by_sampling() {
    DF_CONFIG.set(DataframeConfig::new()).ok();
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
            .unwrap()
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
    .unwrap()
    .collect()
    .unwrap();
    assert_eq!(
        filtered_df["algorithm"],
        Series::new("algorithm", &["algo3".to_string(), "algo3".into()])
    );
}
