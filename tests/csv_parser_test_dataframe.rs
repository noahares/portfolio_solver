use ndarray::{arr1, aview2, Axis};
use portfolio_solver::csv_parser;
use std::path::PathBuf;

#[test]
fn test_dataframe() {
    let files = vec![
        PathBuf::from("data/test/algo1.csv"),
        "data/test/algo2.csv".into(),
    ];
    let k = 2;
    let df = csv_parser::parse_normalized_csvs(&files, None, k).unwrap();
    let data =
        csv_parser::Data::from_normalized_dataframe(df, k, std::f64::MAX)
            .unwrap();
    assert_eq!(data.num_instances, 4);
    assert_eq!(data.num_algorithms, 2);
    assert_eq!(data.best_per_instance, arr1(&[16.0, 7.0, 18.0, 9.0]));
    assert_eq!(
        data.stats.index_axis(Axis(2), 0),
        aview2(&[[18.0, 16.0], [9.0, 7.0], [18.0, 22.0], [9.0, 9.0]])
    );
}
