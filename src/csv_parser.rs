use core::fmt;
use itertools::Itertools;
use polars::{
    lazy::dsl::{Expr, GetOutput},
    prelude::*,
    series::IsSorted,
};
use std::{
    f64::EPSILON,
    path::{Path, PathBuf},
};

use anyhow::Result;

use crate::datastructures::*;

pub use utils::extract_algorithm_columns;

mod utils;

pub struct Data {
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
    pub fn new() -> Result<Self> {
        let Config {
            files: paths,
            graphs: graphs_path,
            ks,
            feasibility_thresholds,
            num_cores: k,
            slowdown_ratio,
            num_seeds: _,
            out_dir: _,
            timeout: _,
        } = Config::global();
        let df_config = DataframeConfig::global();
        let sort_exprs = ["instance", "algorithm", "num_threads"]
            .iter()
            .map(|o| col(o))
            .collect::<Vec<Expr>>();
        let sort_options = vec![false; 3];
        let instance_filter = InstanceFilter {
            instance_path: graphs_path.clone(),
            ks: ks.to_vec(),
            feasibility_thresholds: feasibility_thresholds.to_vec(),
        };
        let df = parse_hypergraph_dataframe(paths, Some(instance_filter), *k)?
            .sort_by_exprs(&sort_exprs, &sort_options, false)
            .collect()?;

        let valid_instance_df = utils::filter_algorithms_by_slowdown(
            df.clone().lazy().filter(col("valid")),
            &["instance"],
            &df_config.algorithm_fields,
            *slowdown_ratio,
        )?
        .sort_by_exprs(&sort_exprs, &sort_options, false)
        .collect()?;

        if valid_instance_df.height() == 0 {
            eprintln!("Error: A portfolio with gmean faster than {slowdown_ratio} * gmean(best) is not possible, try a smaller slowdown ratio.");
            std::process::exit(exitcode::DATAERR);
        }

        let algorithms = utils::extract_algorithm_columns(
            &valid_instance_df,
            &df_config.algorithm_fields,
        )?;
        assert!(algorithms.iter().tuple_windows().all(|(a, b)| a <= b));
        let num_instances = valid_instance_df["instance"].n_unique()?;
        let num_algorithms = algorithms.len();
        let best_per_instance_df = best_per_instance(
            valid_instance_df.clone().lazy(),
            &["instance"],
            "quality",
        )
        .collect()?;
        assert_eq!(
            best_per_instance_df["instance"].is_sorted(),
            IsSorted::Ascending
        );
        let best_per_instance =
            utils::column_to_f64_array(&best_per_instance_df, "best_quality")?;
        assert!(best_per_instance.iter().all(|val| val.abs() >= EPSILON));
        let best_per_instance_time_df = utils::best_per_instance_time(
            valid_instance_df.clone().lazy(),
            &["instance"],
            "quality",
        )
        .collect()?;
        assert_eq!(
            best_per_instance_time_df["instance"].is_sorted(),
            IsSorted::Ascending
        );
        let best_per_instance_time = utils::column_to_f64_array(
            &best_per_instance_time_df,
            "best_time",
        )?;

        let best_per_instance_count = utils::column_to_f64_array(
            &utils::best_per_instance_count(
                valid_instance_df.clone(),
                &["instance"],
                &df_config.algorithm_fields,
                "quality",
            )?,
            "count",
        )?;

        assert_eq!(
            valid_instance_df["instance"].is_sorted(),
            IsSorted::Ascending
        );
        let stats_df = utils::stats_by_sampling(
            valid_instance_df.lazy(),
            *k,
            &["instance"],
            &df_config.algorithm_fields,
        )?
        .collect()?;

        let clean_df = utils::cleanup_missing_rows(
            stats_df,
            *k,
            &["instance"],
            &df_config.algorithm_fields,
        )?
        .lazy()
        .sort_by_exprs(&sort_exprs, &sort_options, false)
        .collect()?;

        assert_eq!(clean_df["instance"].is_sorted(), IsSorted::Ascending);
        let shape = (num_instances, num_algorithms, *k as usize);
        assert_eq!(
            num_instances * num_algorithms * *k as usize,
            clean_df.height()
        );
        let stats: ndarray::Array3<f64> =
            ndarray::Array3::<f64>::from_shape_vec(
                shape,
                clean_df
                    .column("e_min")?
                    .f64()?
                    .into_no_null_iter()
                    .collect::<Vec<f64>>(),
            )?;
        assert_eq!(df["instance"].is_sorted(), IsSorted::Ascending);
        Ok(Self {
            algorithms,
            best_per_instance,
            best_per_instance_time,
            best_per_instance_count,
            stats,
            num_instances,
            num_algorithms,
        })
    }
}

pub struct InstanceFilter {
    instance_path: PathBuf,
    ks: Vec<i64>,
    feasibility_thresholds: Vec<f64>,
}

pub fn parse_hypergraph_dataframe(
    paths: &[PathBuf],
    desired_instances: Option<InstanceFilter>,
    num_cores: u32,
) -> Result<LazyFrame> {
    let schema =
        Schema::from(vec![Field::new("km1", DataType::Float64)].into_iter());
    let instance_fields = vec!["graph", "k", "epsilon"]
        .iter()
        .map(|field| col(field))
        .collect_vec();
    let hypergraph_columns = vec![
        "algorithm".to_string(),
        "num_threads".into(),
        "graph".into(),
        "k".into(),
        "epsilon".into(),
        "imbalance".into(),
        "km1".into(),
        "totalPartitionTime".into(),
        "failed".into(),
        "timeout".into(),
    ];
    let read_df = |path: &PathBuf,
                   in_fields: &Vec<String>|
     -> Result<LazyFrame> {
        let mut dataframe = CsvReader::from_path(path)?
            .with_comment_char(Some(b'#'))
            .has_header(true)
            .with_columns(Some(in_fields.to_vec()))
            .with_dtypes(Some(&schema))
            .finish()?
            .lazy()
            .filter(col("num_threads").lt_eq(lit(num_cores)))
            .with_columns([
                col("graph").apply(
                    |s: Series| {
                        Ok(s.utf8()?
                            .into_no_null_iter()
                            .map(utils::fix_instance_names)
                            .collect())
                    },
                    GetOutput::from_type(DataType::Utf8),
                ),
                col("km1").apply(
                    |s: Series| {
                        Ok(s.f64()?
                            .into_no_null_iter()
                            .map(|i| if i.abs() <= EPSILON { 1.0 } else { i })
                            .collect())
                    },
                    GetOutput::from_type(DataType::Float64),
                ),
            ]);
        match &desired_instances {
            Some(filter) => {
                if let Ok(instance_filter) = utils::get_desired_instances(
                    &filter.instance_path,
                    &filter.ks,
                    &filter.feasibility_thresholds,
                ) {
                    dataframe = dataframe.join(
                        instance_filter,
                        &instance_fields,
                        &instance_fields,
                        JoinType::Inner,
                    );
                }
            }
            None => (),
        };
        Ok(dataframe.select([
            concat_str(&instance_fields, "").alias("instance"),
            col("algorithm"),
            col("num_threads"),
            col("km1").alias("quality"),
            col("totalPartitionTime").alias("time"),
            col("imbalance")
                .lt_eq(col("epsilon"))
                .and(col("failed").eq(lit("no")))
                .and(col("timeout").eq(lit("no")))
                .alias("valid"),
        ]))
    };

    let mut fixed_in_fields = hypergraph_columns.clone();
    fixed_in_fields.retain(|s| s != "num_threads");
    let dataframes: Vec<LazyFrame> = paths
        .iter()
        .map(|path| match read_df(path, &hypergraph_columns) {
            Ok(result) => result,
            Err(_) => read_df(path, &fixed_in_fields)
                .unwrap()
                .with_column(lit(1_i64).alias("num_threads")),
        })
        .collect();
    concat(dataframes, true, true).map_err(anyhow::Error::from)
}

pub fn preprocess_df<T: AsRef<Path>>(
    paths: &[T],
    config: &DataframeConfig,
) -> Result<LazyFrame> {
    let read_df = |path: &T,
                   in_fields: &Vec<String>,
                   out_fields: &Vec<&str>|
     -> Result<LazyFrame> {
        Ok(CsvReader::from_path(path.as_ref())?
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
                        Ok(s.utf8()?
                            .into_no_null_iter()
                            .map(utils::fix_instance_names)
                            .collect())
                    },
                    GetOutput::from_type(DataType::Utf8),
                ),
                col("quality").apply(
                    |s: Series| {
                        Ok(s.f64()?
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
    concat(dataframes, false, false).map_err(anyhow::Error::from)
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
    _df_config: &DataframeConfig,
    path: PathBuf,
) -> Result<()> {
    let mut out = std::fs::File::create(path)?;
    let mut out_df = df.collect()?;
    CsvWriter::new(&mut out)
        .has_header(true)
        .finish(&mut out_df)
        .map_err(anyhow::Error::from)
}

#[cfg(test)]
mod tests;
