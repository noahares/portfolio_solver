use super::utils::{
    best_per_instance_count, filter_algorithms_by_slowdown, stats_by_sampling,
};
use polars::prelude::*;

#[test]
fn test_best_per_instance_count() {
    let df = df! {
            "instance" => ["graph1", "graph1", "graph1", "graph2", "graph2", "graph2"],
            "k" => vec![2; 6],
            "algorithm" => ["algo1", "algo2", "algo3", "algo1", "algo2", "algo3"],
            "num_threads" => vec![1; 6],
            "quality" => [1.0, 2.0, 2.0, 2.0, 1.0, 2.0],
        }.unwrap();
    let ranking = best_per_instance_count(df).unwrap();
    assert_eq!(
        ranking["count"],
        Series::from_vec("count", vec![1.0, 1.0, 0.0])
    );
}

#[test]
fn test_stats_by_sampling() {
    let df = df! {
            "instance" => ["graph1", "graph1", "graph1", "graph1", "graph2", "graph2", "graph2", "graph2"],
            "algorithm" => ["algo1", "algo1", "algo1", "algo1", "algo1", "algo1", "algo1", "algo1"],
            "num_threads" => vec![1; 8],
            "quality" => [10.0, 8.0, 9.0, 7.0, 20.0, 18.0, 22.0, 19.0],
        }.unwrap();
    let stats_df = stats_by_sampling(df.lazy(), 4).unwrap().collect().unwrap();
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
    let df = df! {
            "instance" => ["graph1", "graph1", "graph1", "graph2", "graph2", "graph2"],
            "algorithm" => ["algo1", "algo2", "algo3", "algo1", "algo2", "algo3"],
            "num_threads" => vec![1; 6],
            "quality" => [1.0, 2.0, 2.0, 2.0, 1.0, 2.0],
            "time" => [2.0, 2.0, 1.0, 2.0, 2.0, 1.0],
        }.unwrap();
    let filtered_df = filter_algorithms_by_slowdown(df.lazy(), 0.5)
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(
        filtered_df["algorithm"],
        Series::new("algorithm", &["algo3".to_string(), "algo3".into()])
    );
}
