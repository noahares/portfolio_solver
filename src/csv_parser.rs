use core::fmt;
use itertools::Itertools;
use polars::{
    lazy::dsl::{Expr, GetOutput},
    prelude::*,
    series::IsSorted,
};
use std::f64::EPSILON;

use anyhow::Result;

use crate::datastructures::*;

pub use utils::extract_algorithm_columns;

mod utils;

pub struct Data {
    pub df: DataFrame,
    pub instances: ndarray::Array1<Instance>,
    pub algorithms: ndarray::Array1<Algorithm>,
    pub best_per_instance: ndarray::Array1<f64>,
    pub best_per_instance_time: ndarray::Array1<f64>,
    pub best_per_instance_count: ndarray::Array1<f64>,
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
    pub fn new(config: &Config) -> Self {
        let Config {
            files: paths,
            graphs: graphs_path,
            ks: num_parts,
            feasibility_thresholds,
            num_cores: k,
            slowdown_ratio,
            num_seeds: _,
            out_dir: _,
            timeout: _,
        } = config.clone();
        let df_config = DataframeConfig::new();
        let sort_exprs = df_config
            .sort_order
            .iter()
            .map(|o| col(o))
            .collect::<Vec<Expr>>();
        let sort_options = vec![
            false;
            df_config.instance_fields.len()
                + df_config.algorithm_fields.len()
        ];
        let df = utils::filter_desired_instances(
            preprocess_df(paths.as_ref(), &df_config),
            &graphs_path,
            &num_parts,
            &feasibility_thresholds,
            &df_config.instance_fields,
        )
        .sort_by_exprs(&sort_exprs, &sort_options, false)
        .collect()
        .expect("Failed to collect preprocessed dataframe");

        let valid_instance_df = utils::filter_algorithms_by_slowdown(
            df.clone()
                .lazy()
                .filter(
                    col("feasibility_score")
                        .lt_eq(col("feasibility_threshold")),
                )
                .filter(col("failed").eq(lit("no")))
                .filter(col("timeout").eq(lit("no")))
                .filter(col("num_threads").lt_eq(lit(k))),
            &df_config.instance_fields,
            &df_config.algorithm_fields,
            slowdown_ratio,
        )
        .sort_by_exprs(&sort_exprs, &sort_options, false)
        .collect()
        .expect("Failed to collect valid instances dataframe");

        if valid_instance_df.height() == 0 {
            eprintln!("Error: A portfolio with gmean faster than {slowdown_ratio} * gmean(best) is not possible, try a smaller slowdown ratio.");
            std::process::exit(exitcode::DATAERR);
        }

        let instances = utils::extract_instance_columns(
            &valid_instance_df,
            &df_config.instance_fields,
        );
        assert!(instances.iter().tuple_windows().all(|(a, b)| a <= b));
        let algorithms = utils::extract_algorithm_columns(
            &valid_instance_df,
            &df_config.algorithm_fields,
        );
        assert!(algorithms.iter().tuple_windows().all(|(a, b)| a <= b));
        let num_instances = instances.len();
        let num_algorithms = algorithms.len();
        let best_per_instance_df = best_per_instance(
            valid_instance_df.clone().lazy(),
            &df_config.instance_fields,
            "quality",
        )
        .collect()
        .expect("Failed to collect best_per_instance dataframe");
        assert_eq!(
            best_per_instance_df.column("instance").unwrap().is_sorted(),
            IsSorted::Ascending
        );
        let best_per_instance =
            utils::column_to_f64_array(&best_per_instance_df, "best_quality");
        assert!(best_per_instance.iter().all(|val| val.abs() >= EPSILON));
        let best_per_instance_time_df = utils::best_per_instance_time(
            valid_instance_df.clone().lazy(),
            &df_config.instance_fields,
            "quality",
        )
        .collect()
        .expect("Failed to collect best_per_instance_time dataframe");
        assert_eq!(
            best_per_instance_time_df
                .column("instance")
                .unwrap()
                .is_sorted(),
            IsSorted::Ascending
        );
        let best_per_instance_time = utils::column_to_f64_array(
            &best_per_instance_time_df,
            "best_time",
        );

        let best_per_instance_count = utils::column_to_f64_array(
            &utils::best_per_instance_count(
                valid_instance_df.clone(),
                &df_config.instance_fields,
                &df_config.algorithm_fields,
                "quality",
            ),
            "count",
        );

        assert_eq!(
            valid_instance_df.column("instance").unwrap().is_sorted(),
            IsSorted::Ascending
        );
        let stats_df = utils::stats_by_sampling(
            valid_instance_df.lazy(),
            k,
            &df_config.instance_fields,
            &df_config.algorithm_fields,
        )
        .collect()
        .expect("Failed to collect stats dataframe");

        let clean_df = utils::cleanup_missing_rows(
            stats_df,
            k,
            &df_config.instance_fields,
            &df_config.algorithm_fields,
        )
        .lazy()
        .sort_by_exprs(&sort_exprs, &sort_options, false)
        .collect()
        .unwrap();

        assert_eq!(
            clean_df.column("instance").unwrap().is_sorted(),
            IsSorted::Ascending
        );
        let shape = (num_instances, num_algorithms, k as usize);
        assert_eq!(
            num_instances * num_algorithms * k as usize,
            clean_df.height()
        );
        let stats: ndarray::Array3<f64> =
            ndarray::Array3::<f64>::from_shape_vec(
                shape,
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
            best_per_instance_count,
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
                            .map(utils::fix_instance_names)
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
            ]))
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
}

pub fn best_per_instance(
    df: LazyFrame,
    instance_fields: &[&str],
    target_field: &str,
) -> LazyFrame {
    df.groupby_stable(instance_fields)
        .agg([min(target_field).prefix("best_")])
}

pub fn df_to_csv_for_performance_profiles(
    df: LazyFrame,
    df_config: &DataframeConfig,
    path: &str,
) {
    let mut out =
        std::fs::File::create(path).expect("Failed to create output file");
    let mut out_df = df
        .rename(&df_config.out_fields, &df_config.in_fields)
        .collect()
        .expect("Missmatching fields for output dataframe");
    CsvWriter::new(&mut out)
        .has_header(true)
        .finish(&mut out_df)
        .expect("Failed to write output file");
}

#[cfg(test)]
mod tests;
