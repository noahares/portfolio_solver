use super::{get_a_start, round_to_sum, solve};
use crate::{csv_parser::Data, datastructures::*, test_utils::*};

#[test]
fn test_simple_model() {
    let config = Config {
        files: vec![
            "data/test/algo1.csv".to_string(),
            "data/test/algo2.csv".into(),
        ],
        ..default_config()
    };
    let k = config.num_cores;
    let data = Data::new(&config);
    assert_eq!(
        solve(&data, k as usize, Timeout::default()),
        SolverResult {
            resource_assignments: vec![
                (
                    Algorithm {
                        algorithm: "algo1".into(),
                        num_threads: 1
                    },
                    1.0
                ),
                (
                    Algorithm {
                        algorithm: "algo2".into(),
                        num_threads: 1
                    },
                    1.0
                ),
            ]
        }
    );
}

#[test]
fn test_seq_vs_par() {
    let config = Config {
        files: vec![
            "data/test/algo1.csv".to_string(),
            "data/test/algo7.csv".into(),
        ],
        num_cores: 8,
        ..default_config()
    };
    let k = config.num_cores;
    let data = Data::new(&config);
    assert_eq!(
        solve(&data, k as usize, Timeout::default()),
        SolverResult {
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

#[test]
fn test_a_start_values() {
    let stats =
        ndarray::array![[[1.0, 2.0], [3.0, 4.0]], [[7.0, 6.0], [5.0, 8.0]]];
    assert_eq!(get_a_start(&stats, 2), vec![(0, 0, 0), (1, 1, 0)]);
}

#[test]
fn test_round_to_sum() {
    let fractions = vec![2.4, 1.6, 0.8, 1.9, 1.6];
    let steps = vec![1, 2, 4, 8, 1];
    let sum = 20;
    assert_eq!(
        round_to_sum(&fractions, &steps, sum),
        vec![2.0, 2.0, 1.0, 1.0, 2.0]
    );
}