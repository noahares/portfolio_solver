use ndarray::arr1;
use portfolio_solver::csv_parser;
use std::path::PathBuf;

#[test]
fn test_missing_algo_for_instance() {
    let files = vec![
        PathBuf::from("data/test/algo2.csv"),
        "data/test/algo5.csv".into(),
    ];
    let k = 2;
    let df = csv_parser::parse_normalized_csvs(&files, None, k).unwrap();
    let data =
        csv_parser::Data::from_normalized_dataframe(df, k, std::f64::MAX)
            .unwrap();
    assert_eq!(data.num_instances, 4);
    assert_eq!(data.num_algorithms, 2);
    assert_eq!(data.best_per_instance, arr1(&[16.0, 7.0, 22.0, 9.0]));
}
