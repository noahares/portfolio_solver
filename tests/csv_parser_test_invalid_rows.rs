use ndarray::arr1;
use portfolio_solver::csv_parser;
use std::path::PathBuf;

#[test]
fn test_invalid_rows() {
    let files = vec![PathBuf::from("data/test/algo6.csv")];
    let k = 2;
    let df = csv_parser::parse_normalized_csvs(&files, None, k).unwrap();
    let data =
        csv_parser::Data::from_normalized_dataframe(df, k, std::f64::MAX)
            .unwrap();
    assert_eq!(data.num_instances, 4);
    assert_eq!(data.num_algorithms, 1);
    assert_eq!(data.best_per_instance, arr1(&[20.0, 20.0, 20.0, 20.0]));
}
