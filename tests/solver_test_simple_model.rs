use portfolio_solver::{csv_parser, datastructures::*, solver::solve};
use std::path::PathBuf;

#[test]
fn test_simple_model() {
    let files = vec![
        PathBuf::from("data/test/algo1.csv"),
        "data/test/algo2.csv".into(),
    ];
    let k = 2;
    let df = csv_parser::parse_normalized_csvs(&files, None, k).unwrap();
    let data =
        csv_parser::Data::from_normalized_dataframe(df, k, std::f64::MAX)
            .unwrap();
    assert_eq!(
        solve(&data, k as usize, Timeout::default(), None)
            .unwrap()
            .final_portfolio,
        Portfolio {
            name: "final_portfolio_opt".to_string(),
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
