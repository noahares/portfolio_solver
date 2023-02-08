use core::fmt;
use itertools::{izip, Itertools};
use polars::{
    lazy::dsl::{as_struct, Expr, GetOutput},
    prelude::*,
    series::IsSorted,
};
use std::f64::EPSILON;

use anyhow::Result;

use crate::datastructures::*;

pub struct Data {
    pub df: DataFrame,
    pub instances: ndarray::Array1<Instance>,
    pub algorithms: ndarray::Array1<Algorithm>,
    pub best_per_instance: ndarray::Array1<f64>,
    pub best_per_instance_time: ndarray::Array1<f64>,
    pub stats: ndarray::Array3<f64>,
    pub num_instances: usize,
    pub num_algorithms: usize,
}

impl fmt::Display for Data {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "m: {}, n: {}", self.num_instances, self.num_algorithms)
    }
}

impl Data {
    pub fn new(config: Config) -> Self {
        let Config {
            files: paths,
            quality_lb: quality_lb_path,
            num_cores: k,
            slowdown_ratio,
            num_seeds: _,
            out_file: _,
        } = config;
        let df_config = DataframeConfig::new();
        let df = preprocess_df(paths.as_ref(), &df_config)
            .collect()
            .expect("Failed to collect preprocessed dataframe");

        let instances =
            extract_instance_columns(&df, &df_config.instance_fields);
        let algorithms =
            extract_algorithm_columns(&df, &df_config.algorithm_fields);
        let num_instances = instances.len();
        let num_algorithms = algorithms.len();
        let best_per_instance_df = best_per_instance(
            df.clone().lazy(),
            &df_config.instance_fields,
            "quality",
        )
        .collect()
        .expect("Failed to collect best_per_instance dataframe");
        let best_per_instance =
            column_to_f64_array(&best_per_instance_df, "best_quality");
        assert!(best_per_instance.iter().all(|val| val.abs() >= EPSILON));
        let best_per_instance_time_df = best_per_instance_time(
            df.clone().lazy(),
            &df_config.instance_fields,
            "quality",
        );
        let best_per_instance_time = column_to_f64_array(
            &best_per_instance_time_df
                .clone()
                .collect()
                .expect("Failed to collect best_per_instance_time dataframe"),
            "best_time",
        );
        let slowdown_ratio_df = filter_slowdown(
            df.clone().lazy(),
            &df_config.instance_fields,
            slowdown_ratio,
            best_per_instance_time_df,
        )
        .collect()
        .expect("Failed to collect slowdown_ratio dataframe");

        let stats_df = stats(
            slowdown_ratio_df.lazy(),
            k,
            &df_config.instance_fields,
            &df_config.algorithm_fields,
            quality_lb_path,
        )
        .expect(
            "Something went very wrong, stats dataframe could not be created",
        )
        .collect()
        .expect("Failed to collect stats dataframe");

        let clean_df = cleanup_missing_rows(
            stats_df,
            k,
            &df_config.instance_fields,
            &df_config.algorithm_fields,
        );

        let stats: ndarray::Array3<f64> =
            ndarray::Array3::<f64>::from_shape_vec(
                (num_instances, num_algorithms, k as usize),
                clean_df
                    .column("e_min")
                    .expect("Something went very wrong, no `e_min` column")
                    .f64()
                    .expect("Something went very wrong, `e_min` column has unexpected type")
                    .into_no_null_iter()
                    .collect::<Vec<f64>>(),
            )
            .expect("Failed to create stats array");
        assert_eq!(
            df.column("instance").unwrap().is_sorted(),
            IsSorted::Ascending
        );
        Self {
            df,
            instances,
            algorithms,
            best_per_instance,
            best_per_instance_time,
            stats,
            num_instances,
            num_algorithms,
        }
    }
}

pub fn preprocess_df<T: AsRef<str>>(
    paths: &[T],
    config: &DataframeConfig,
) -> LazyFrame {
    let read_df = |path: &T,
                   in_fields: &Vec<String>,
                   out_fields: &Vec<&str>|
     -> Result<LazyFrame> {
        Ok(CsvReader::from_path(path.as_ref())
            .unwrap_or_else(|_| {
                panic!("No csv file found under {}", path.as_ref())
            })
            .with_comment_char(Some(b'#'))
            .has_header(true)
            .with_columns(Some(in_fields.to_vec()))
            .with_dtypes(Some(&config.schema))
            .finish()?
            .lazy()
            .rename(in_fields.iter(), out_fields.iter())
            .with_columns([
                col("instance").apply(
                    |s: Series| {
                        Ok(s.utf8()
                            .expect("Field `instance` should be a string")
                            .into_no_null_iter()
                            .map(fix_instance_names)
                            .collect())
                    },
                    GetOutput::from_type(DataType::Utf8),
                ),
                col("quality").apply(
                    |s: Series| {
                        Ok(s.f64()
                            .expect("Field `quality` should be a float")
                            .into_no_null_iter()
                            .map(|i| if i.abs() <= EPSILON { 1.0 } else { i })
                            .collect())
                    },
                    GetOutput::from_type(DataType::Float64),
                ),
            ])
            .filter(col("feasibility_score").lt_eq(0.03))
            .filter(col("failed").str().contains("no"))
            .filter(col("timeout").str().contains("no")))
    };

    let mut fixed_in_fields = config.in_fields.clone();
    fixed_in_fields.retain(|s| s != "num_threads");
    let mut fixed_out_fields = config.out_fields.clone();
    fixed_out_fields.retain(|&s| s != "num_threads");
    let dataframes: Vec<LazyFrame> = paths
        .iter()
        .map(|path| {
            match read_df(path, &config.in_fields, &config.out_fields) {
                Ok(result) => result,
                Err(_) => read_df(path, &fixed_in_fields, &fixed_out_fields)
                    .unwrap()
                    .with_column(lit(1_i64).alias("num_threads")),
            }
            .select(
                &config
                    .out_fields
                    .iter()
                    .map(|c| col(c))
                    .collect::<Vec<Expr>>(),
            )
        })
        .collect();
    concat(dataframes, false, false)
        .expect("Combining data from csv files failed")
        .sort_by_exprs(
            config
                .sort_order
                .iter()
                .map(|o| col(o))
                .collect::<Vec<Expr>>(),
            vec![
                false;
                config.instance_fields.len() + config.algorithm_fields.len()
            ],
            false,
        )
}

fn read_quality_lb(path: String, instance_fields: &[&str]) -> DataFrame {
    CsvReader::from_path(&path)
       .unwrap_or_else(|_| {
           panic!("No csv file found under {}, generate it with the dedicated binary", path)
       })
       .with_comment_char(Some(b'#'))
       .has_header(true)
       .with_columns(Some([instance_fields.iter().map(|s| s.to_string()).collect_vec(), vec!["quality_lb".to_string()]].concat()))
       .finish()
       .expect("Failed to read lower bound csv file")
}

fn fix_instance_names(instance: &str) -> String {
    if instance.ends_with("scotch") {
        instance.replace("scotch", "graph")
    } else if instance.ends_with("mtx") {
        instance.replace("mtx", "mtx.hgr")
    } else {
        instance.to_string()
    }
}

fn extract_instance_columns(
    df: &DataFrame,
    instance_fields: &[&str],
) -> ndarray::Array1<Instance> {
    let unique_instances_df = df
        .clone()
        .lazy()
        .unique_stable(
            Some(instance_fields.iter().map(|s| s.to_string()).collect_vec()),
            UniqueKeepStrategy::First,
        )
        .collect()
        .expect("Failed to extract instance columns");
    let instance_it = unique_instances_df
        .column("instance")
        .expect("No field `instance`")
        .utf8()
        .expect("Field `instance` should be a string")
        .into_no_null_iter();
    let k_it = unique_instances_df
        .column("k")
        .expect("No field `k`")
        .i64()
        .expect("Field `k` should be a integer")
        .into_no_null_iter();
    ndarray::Array1::from_iter(
        instance_it
            .zip(k_it)
            .map(|(i, k)| Instance::new(i.to_string(), k as u32)),
    )
}

fn extract_algorithm_columns(
    df: &DataFrame,
    algorithm_fields: &[&str],
) -> ndarray::Array1<Algorithm> {
    let unique_algorithm_df = df
        .clone()
        .lazy()
        .unique_stable(
            Some(algorithm_fields.iter().map(|s| s.to_string()).collect_vec()),
            UniqueKeepStrategy::First,
        )
        .collect()
        .expect("Failed to extract instance columns");
    let algorithm_it = unique_algorithm_df
        .column("algorithm")
        .expect("No field `algorithm`")
        .utf8()
        .expect("Field `algorithm` should be a string")
        .into_no_null_iter();
    let num_threads = unique_algorithm_df
        .column("num_threads")
        .expect("No field `num_threads`")
        .i64()
        .expect("Field `num_threads` should be a integer")
        .into_no_null_iter();
    ndarray::Array1::from_iter(
        algorithm_it
            .zip(num_threads)
            .map(|(a, t)| Algorithm::new(a.to_string(), t as u32)),
    )
}

pub fn best_per_instance(
    df: LazyFrame,
    instance_fields: &[&str],
    target_field: &str,
) -> LazyFrame {
    df.groupby_stable(instance_fields)
        .agg([min(target_field).prefix("best_")])
}

fn best_per_instance_time(
    df: LazyFrame,
    instance_fields: &[&str],
    target_field: &str,
) -> LazyFrame {
    df.groupby_stable(instance_fields)
        .agg([col("*")
            .sort_by(vec![col(target_field)], vec![false])
            .first()])
        .rename(["time"], ["best_time"])
        .select(
            [
                instance_fields.iter().map(|field| col(field)).collect_vec(),
                vec![col("best_time")],
            ]
            .concat(),
        )
}

fn column_to_f64_array(
    df: &DataFrame,
    column_name: &str,
) -> ndarray::Array1<f64> {
    df.column(column_name)
        .unwrap_or_else(|_| panic!("Field `{}` not found", column_name))
        .f64()
        .expect("Expected column with type float")
        .to_ndarray()
        .unwrap_or_else(|_| {
            panic!("Failed to extract column `{}`", column_name)
        })
        .to_owned()
}

fn stats(
    df: LazyFrame,
    sample_size: u32,
    instance_fields: &[&str],
    algorithm_fields: &[&str],
    quality_lb_path: String,
) -> Result<LazyFrame> {
    let quality_lb = read_quality_lb(quality_lb_path, instance_fields);
    let possible_repeats = df! {
        "sample_size" => Vec::from_iter(1..=sample_size)
    }
    .unwrap();

    let columns = [instance_fields, algorithm_fields, &["sample_size"]]
        .concat()
        .iter()
        .map(|f| col(f))
        .collect_vec();
    Ok(df
        .groupby_stable([algorithm_fields, instance_fields].concat())
        .agg([
            col("quality")
                .filter(col("quality").lt(lit(std::f64::MAX)))
                .mean()
                .fill_null(lit(std::f64::MAX))
                .prefix("mean_"),
            col("quality")
                .filter(col("quality").lt(lit(std::f64::MAX)))
                .std(1)
                .fill_null(lit(std::f64::MAX))
                .prefix("std_"),
        ])
        .join(
            quality_lb.lazy(),
            instance_fields.iter().map(|field| col(field)).collect_vec(),
            instance_fields.iter().map(|field| col(field)).collect_vec(),
            JoinType::Inner,
        )
        .cross_join(possible_repeats.lazy())
        .select(
            [
                columns,
                [as_struct(&[
                    col("mean_quality"),
                    col("std_quality"),
                    col("sample_size"),
                    col("quality_lb"),
                ])
                .apply(
                    |s| {
                        let data = s.struct_()?;
                        let (
                            mean_series,
                            std_series,
                            sample_series,
                            min_series,
                        ) = (
                            &data.fields()[0].f64()?,
                            &data.fields()[1].f64()?,
                            &data.fields()[2].u32()?,
                            &data.fields()[3].f64()?,
                        );
                        let result: Float64Chunked = izip!(
                            mean_series.into_iter(),
                            std_series.into_iter(),
                            sample_series.into_iter(),
                            min_series.into_iter()
                        )
                        .map(|(opt_mean, opt_std, opt_s, opt_min)| {
                            match (opt_mean, opt_std, opt_s, opt_min) {
                                (
                                    Some(mean),
                                    Some(std),
                                    Some(s),
                                    Some(min),
                                ) => Some(expected_normdist_min(
                                    mean, std, s, min,
                                )),
                                _ => None,
                            }
                        })
                        .collect();
                        Ok(result.into_series())
                    },
                    GetOutput::from_type(DataType::Float64),
                )
                .alias("e_min")]
                .to_vec(),
            ]
            .concat(),
        ))
}

fn expected_normdist_min(
    mean: f64,
    std: f64,
    sample_size: u32,
    min: f64,
) -> f64 {
    let result =
        (mean - std * expected_maximum_approximation(sample_size)).max(min);
    assert!(result >= 0.0, "mean: {mean}, std: {std}, e_min: {result}");
    result
}

fn expected_maximum_approximation(sample_size: u32) -> f64 {
    f64::sqrt(2.0 * f64::ln(sample_size as f64))
}

fn cleanup_missing_rows(
    df: DataFrame,
    k: u32,
    instance_fields: &[&str],
    algorithm_fields: &[&str],
) -> DataFrame {
    let algorithm_series = df
        .select(algorithm_fields)
        .expect("Cleanup failed due to `algorithm_fields`")
        .unique_stable(None, UniqueKeepStrategy::First)
        .expect("Cleanup failed due to `algorithm_fields`")
        .lazy();
    let instance_series = df
        .select(instance_fields)
        .expect("Cleanup failed due to `instance_fields`")
        .unique_stable(None, UniqueKeepStrategy::First)
        .expect("Cleanup failed due to `instance_fields`")
        .lazy();
    let possible_repeats = df! {
        "sample_size" => Vec::from_iter(1..=k)
    }
    .unwrap();
    let full_df = instance_series
        .cross_join(algorithm_series)
        .cross_join(possible_repeats.lazy())
        .collect()
        .expect("Failed to create carthesian product dataframe");
    let target_columns =
        [instance_fields, algorithm_fields, &["sample_size"]].concat();
    df.outer_join(&full_df, &target_columns, &target_columns)
        .expect("Failed to fill missing rows")
        .fill_null(FillNullStrategy::MaxBound)
        .expect("Failed to fix null values")
}

fn filter_slowdown(
    df: LazyFrame,
    instance_fields: &[&str],
    slowdown_ratio: f64,
    best_per_instance_time_df: LazyFrame,
) -> LazyFrame {
    df.join(
        best_per_instance_time_df,
        instance_fields.iter().map(|field| col(field)).collect_vec(),
        instance_fields.iter().map(|field| col(field)).collect_vec(),
        JoinType::Inner,
    ) // .with_column(
      //     when(col("time").lt(col("best_time") * lit(slowdown_ratio))).then(col("quality")).otherwise(std::f64::MAX).alias("quality"))
      // .filter(col("time").lt(col("best_time") * lit(slowdown_ratio)))
}

pub fn df_to_csv_for_performance_profiles(
    df: &DataFrame,
    df_config: &DataframeConfig,
    path: &str,
) {
    let mut out =
        std::fs::File::create(path).expect("Failed to create output file");
    let mut out_df = df
        .clone()
        .lazy()
        .rename(&df_config.out_fields, &df_config.in_fields)
        .collect()
        .expect("Missmatching fields for output dataframe");
    CsvWriter::new(&mut out)
        .has_header(true)
        .finish(&mut out_df)
        .expect("Failed to write output file");
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, aview2, Axis};

    use crate::{csv_parser::Data, datastructures::Config};

    fn default_config() -> Config {
        Config {
            files: vec![],
            quality_lb: "data/test/quality_lb.csv".to_string(),
            num_cores: 2,
            slowdown_ratio: std::f64::MAX,
            num_seeds: 1,
            out_file: "".to_string(),
        }
    }

    #[test]
    fn test_dataframe() {
        let config = Config {
            files: vec![
                "data/test/algo1.csv".to_string(),
                "data/test/algo2.csv".into(),
            ],
            ..default_config()
        };
        let data = Data::new(config);
        assert_eq!(data.num_instances, 4);
        assert_eq!(data.num_algorithms, 2);
        assert_eq!(data.best_per_instance, arr1(&[16.0, 7.0, 18.0, 9.0]));
        assert_eq!(
            data.stats.index_axis(Axis(2), 0),
            aview2(&[[20.0, 18.0], [10.0, 8.0], [20.0, 24.0], [10.0, 11.0]])
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
        let data = Data::new(config);
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
        let data = Data::new(config);
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
        let data = Data::new(config);
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
        let data = Data::new(config);
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
        let data = Data::new(config);
        assert_eq!(data.num_instances, 4);
        assert_eq!(data.num_algorithms, 2);
        assert_eq!(data.best_per_instance_time, arr1(&[1.2, 4.2, 2.0, 3.0]));
    }
}
