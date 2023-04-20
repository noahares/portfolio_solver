use core::fmt;
use itertools::Itertools;
use polars::{prelude::*, series::IsSorted};
use std::{f64::EPSILON, path::PathBuf};

use anyhow::Result;

use crate::datastructures::*;

pub use utils::extract_algorithm_columns;

mod utils;

/// Input data structure for the solver, parser for nomalized data frame is available.
pub struct Data {
    /// A list of algorithms to consider for the portfolio
    pub algorithms: ndarray::Array1<Algorithm>,
    /// Contains the best quality for each instance
    pub best_per_instance: ndarray::Array1<f64>,
    /// Contains the number of instances for each algorithm where it is the best,
    /// used as heuristic for a initial solution for [`crate::solver::solve`] if none is provided explicitly
    pub best_per_instance_count: Option<ndarray::Array1<f64>>,
    /// Expected best quality for each instance for each algorithm for each possible number of
    /// repetitions.
    ///
    /// Dimension 1: Instance,
    ///
    /// Dimension 2: Algorithm,
    ///
    /// Dimension 3: Repetitions
    pub expected_best_quality: ndarray::Array3<f64>,
    /// number of instances
    pub num_instances: usize,
    /// number of algorithms
    pub num_algorithms: usize,
}

impl fmt::Display for Data {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "m: {}, n: {}", self.num_instances, self.num_algorithms)
    }
}

impl Data {
    /// Create a new set of input data for [`crate::solver::solve`] from existing data.
    /// This method is **not** advised, since order is very important here.
    /// Once some refactoring is done, this will be easier.
    pub fn new(
        algorithms: &[Algorithm],
        best_per_instance: &[f64],
        best_per_instance_count: Option<&[f64]>,
        stats: &[f64],
        k: u32,
    ) -> Result<Self> {
        let num_algorithms = algorithms.len();
        let num_instances = best_per_instance.len();
        let best_per_instance_count = best_per_instance_count
            .map(|iter| ndarray::Array1::from_iter(iter.to_vec()));
        let shape = (num_instances, num_algorithms, k as usize);
        Ok(Self {
            algorithms: ndarray::Array1::from_iter(algorithms.to_vec()),
            best_per_instance: ndarray::Array1::from_iter(
                best_per_instance.to_vec(),
            ),
            best_per_instance_count,
            expected_best_quality: ndarray::Array3::from_shape_vec(
                shape,
                stats.to_vec(),
            )?,
            num_instances,
            num_algorithms,
        })
    }

    /// Create a new set of input data for [`crate::solver::solve`] from a normalized data frame
    pub fn from_normalized_dataframe(
        df: LazyFrame,
        k: u32,
        slowdown_ratio: f64,
    ) -> Result<Self> {
        let sort_exprs: [Expr; 3] =
            [col("instance"), col("algorithm"), col("num_threads")];
        let sort_options = vec![false; sort_exprs.len()];

        let valid_instance_df = utils::filter_algorithms_by_slowdown(
            df.filter(col("valid")),
            slowdown_ratio,
        )?
        .sort_by_exprs(&sort_exprs, &sort_options, false)
        .collect()?;

        if valid_instance_df.height() == 0 {
            eprintln!("Error: A portfolio with gmean faster than {slowdown_ratio} * gmean(best) is not possible, try a smaller slowdown ratio.");
            std::process::exit(exitcode::DATAERR);
        }

        let algorithms = utils::extract_algorithm_columns(&valid_instance_df)?;
        assert!(algorithms.iter().tuple_windows().all(|(a, b)| a <= b));
        let num_instances = valid_instance_df["instance"].n_unique()?;
        let num_algorithms = algorithms.len();
        let best_per_instance_df = utils::best_per_instance(
            valid_instance_df.clone().lazy(),
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
        let best_per_instance_time_df =
            utils::best_per_instance_time(valid_instance_df.clone().lazy())
                .collect()?;
        assert_eq!(
            best_per_instance_time_df["instance"].is_sorted(),
            IsSorted::Ascending
        );

        let best_per_instance_count = utils::column_to_f64_array(
            &utils::best_per_instance_count(valid_instance_df.clone())?,
            "count",
        )?;

        assert_eq!(
            valid_instance_df["instance"].is_sorted(),
            IsSorted::Ascending
        );
        let stats_df = utils::stats_by_sampling(valid_instance_df.lazy(), k)?
            .collect()?;

        let clean_df = utils::cleanup_missing_rows(stats_df, k)?
            .lazy()
            .sort_by_exprs(&sort_exprs, &sort_options, false)
            .collect()?;

        assert_eq!(clean_df["instance"].is_sorted(), IsSorted::Ascending);
        let shape = (num_instances, num_algorithms, k as usize);
        assert_eq!(
            num_instances * num_algorithms * k as usize,
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
        Ok(Self {
            algorithms,
            best_per_instance,
            best_per_instance_count: Some(best_per_instance_count),
            expected_best_quality: stats,
            num_instances,
            num_algorithms,
        })
    }
}

/// Read normalized data from multiple input files.
///
/// Optionally, provide a path to a csv containing one column `instance` with instances to filter
/// for.
///
/// Normalized csvs have the following header (types in parenthesis):
///
/// algorithm(str),num_threads(int),instance(str),quality(float),time(float),valid(bool)
/// ```csv
/// algorithm,num_threads,instance,quality,time,valid
/// algo1,1,instance1,42.0,10.0,true
/// algo1,1,instance2,40.0,10.5,false
/// algo2,1,instance1,45.0,10.7,true
/// algo2,2,instance2,41.0,10.3,true
/// algo3,1,instance1,42.0,10.0,true
/// ```
pub fn parse_normalized_csvs(
    paths: &[PathBuf],
    desired_instances: Option<PathBuf>,
    num_cores: u32,
) -> Result<LazyFrame> {
    let read_df =
        |path: &PathBuf, in_fields: &[&'static str]| -> Result<LazyFrame> {
            let mut dataframe = CsvReader::from_path(path)?
                .with_comment_char(Some(b'#'))
                .has_header(true)
                .with_columns(Some(
                    in_fields.iter().map(|s| s.to_string()).collect_vec(),
                ))
                .with_dtypes(Some(&Schema::from(
                    [Field::new("quality", DataType::Float64)].into_iter(),
                )))
                .finish()?
                .lazy()
                .filter(col("num_threads").lt_eq(lit(num_cores)))
                .with_columns([col("quality").apply(
                    |s: Series| {
                        Ok(s.f64()?
                            .into_no_null_iter()
                            .map(|i| if i.abs() <= EPSILON { 1.0 } else { i })
                            .collect())
                    },
                    GetOutput::from_type(DataType::Float64),
                )]);
            match &desired_instances {
                Some(filter) => {
                    if let Ok(instance_filter) =
                        utils::get_desired_instances(filter)
                    {
                        dataframe = dataframe.join(
                            instance_filter,
                            &[col("instance")],
                            &[col("instance")],
                            JoinType::Inner,
                        );
                    }
                }
                None => (),
            };
            Ok(dataframe)
        };

    let columns: [&str; 6] = [
        "algorithm",
        "num_threads",
        "instance",
        "quality",
        "time",
        "valid",
    ];
    let dataframes: Vec<LazyFrame> = paths
        .iter()
        .map(|path| read_df(path, &columns))
        .filter_map(Result::ok)
        .collect_vec();
    concat(dataframes, true, true).map_err(anyhow::Error::from)
}

/// Helper to write a data frame to a file
pub fn df_to_normalized_csv(df: LazyFrame, path: PathBuf) -> Result<()> {
    let mut out = std::fs::File::create(path)?;
    let mut out_df = df.collect()?;
    CsvWriter::new(&mut out)
        .has_header(true)
        .finish(&mut out_df)
        .map_err(anyhow::Error::from)
}

#[cfg(test)]
mod tests;
