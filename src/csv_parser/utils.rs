use std::path::PathBuf;

use log::warn;
use polars::prelude::*;

use anyhow::{Context, Result};

use crate::datastructures::*;

/// Get a list of algorithms from the columns of a normalized data frame
///
/// The data frame must contain a string column `algorithm` and a integer column `num_threads`
pub fn extract_algorithm_columns(
    df: &DataFrame,
) -> Result<ndarray::Array1<Algorithm>> {
    let unique_algorithm_df = df
        .clone()
        .lazy()
        .unique_stable(
            Some(vec![String::from("algorithm"), String::from("num_threads")]),
            UniqueKeepStrategy::First,
        )
        .collect()?;
    let algorithm_it = unique_algorithm_df
        .column("algorithm")?
        .utf8()?
        .into_no_null_iter();
    let num_threads = unique_algorithm_df
        .column("num_threads")?
        .i64()?
        .into_no_null_iter();
    Ok(ndarray::Array1::from_iter(
        algorithm_it
            .zip(num_threads)
            .map(|(a, t)| Algorithm::new(a.to_string(), t as u32)),
    ))
}

pub fn best_per_instance(df: LazyFrame, target_field: &str) -> LazyFrame {
    df.groupby_stable(["instance"])
        .agg([min(target_field).prefix("best_")])
}

pub fn best_per_instance_time(df: LazyFrame) -> LazyFrame {
    df.groupby_stable(["instance"])
        .agg([col("*").sort_by(vec![col("quality")], vec![false]).first()])
        .rename(["time"], ["best_time"])
        .select([col("instance"), col("best_time")])
}

pub fn column_to_f64_array(
    df: &DataFrame,
    column_name: &str,
) -> Result<ndarray::Array1<f64>> {
    Ok(df.column(column_name)?.f64()?.to_ndarray()?.to_owned())
}

pub fn stats_by_sampling(
    df: LazyFrame,
    sample_size: u32,
) -> Result<LazyFrame> {
    let columns = vec![col("instance"), col("algorithm"), col("num_threads")];

    let sort_exprs = [columns.clone(), vec![col("sample_size")]].concat();
    let sort_options = vec![false; sort_exprs.len()];
    let samples_per_repeats: Vec<LazyFrame> = (1_u64..=sample_size as u64)
        .map(|s| {
            df.clone()
                .groupby(&columns)
                .agg([col("quality")
                    .sample_n(s as usize, true, true, Some(s))
                    .min()
                    .alias("e_min")])
                .with_column(lit(s as u32).alias("sample_size"))
        })
        .collect();
    Ok(concat(samples_per_repeats, false, false)?.sort_by_exprs(
        &sort_exprs,
        sort_options,
        false,
    ))
}

pub fn cleanup_missing_rows(df: DataFrame, k: u32) -> Result<DataFrame> {
    let algorithm_fields = [col("algorithm"), col("num_threads")];
    let algorithm_series = df
        .clone()
        .lazy()
        .select(&algorithm_fields)
        .unique_stable(None, UniqueKeepStrategy::First);
    let instance_series = df
        .clone()
        .lazy()
        .select([col("instance")])
        .unique_stable(None, UniqueKeepStrategy::First);
    let possible_repeats = df! {
        "sample_size" => Vec::from_iter(1..=k)
    }
    .unwrap();
    let full_df = instance_series
        .cross_join(algorithm_series)
        .cross_join(possible_repeats.lazy())
        .collect()?;
    let columns = [
        vec![col("instance")],
        algorithm_fields.to_vec(),
        vec![col("sample_size")],
    ]
    .concat();
    Ok(df
        .lazy()
        .join(full_df.lazy(), &columns, &columns, JoinType::Outer)
        .collect()?
        .fill_null(FillNullStrategy::MaxBound)?)
}

pub fn filter_algorithms_by_slowdown(
    df: LazyFrame,
    slowdown_ratio: f64,
) -> Result<LazyFrame> {
    let algorithm_fields = [col("algorithm"), col("num_threads")];
    let gmean = |s: Series| -> Result<Series, PolarsError> {
        let gmean = s.f64()?.into_no_null_iter().map(|v| v.ln()).sum::<f64>()
            / s.len() as f64;
        Ok(Series::new("gmean", &[gmean]))
    };
    let best_per_instance_time_df = best_per_instance_time(df.clone());
    let gmean_best_per_instance = {
        let mut gmean_best_per_instance = best_per_instance_time_df
            .select([col("best_time")
                .apply(gmean, GetOutput::from_type(DataType::Float64))
                .alias("gmean")])
            .collect()?
            .column("gmean")?
            .f64()?
            .into_no_null_iter()
            .last()
            .context("empty dataframe")?;
        if gmean_best_per_instance < std::f64::EPSILON {
            warn!("gmean of best algorithms per instance is {}. Setting it to 1.0", gmean_best_per_instance);
            gmean_best_per_instance = 1.0;
        }
        gmean_best_per_instance
    };
    let filtered_df = df
        .clone()
        .groupby(&algorithm_fields)
        .agg([col("time")
            .apply(gmean, GetOutput::from_type(DataType::Float64))
            .first()
            .alias("gmean")])
        .filter(
            col("gmean").lt(lit(slowdown_ratio * gmean_best_per_instance)),
        );
    Ok(df.join(
        filtered_df,
        &algorithm_fields,
        &algorithm_fields,
        JoinType::Inner,
    ))
}

pub fn best_per_instance_count(df: DataFrame) -> Result<DataFrame> {
    let algorithm_fields = [col("algorithm"), col("num_threads")];
    let algorithm_series = df
        .clone()
        .lazy()
        .select(&algorithm_fields)
        .unique_stable(None, UniqueKeepStrategy::First);
    Ok(df
        .lazy()
        .groupby_stable(["instance"])
        .agg([col("*").sort_by(vec![col("quality")], vec![false]).first()])
        .select(&algorithm_fields)
        .groupby_stable(&algorithm_fields)
        .agg([col("*"), count().alias("count").cast(DataType::Float64)])
        .join(
            algorithm_series,
            &algorithm_fields,
            &algorithm_fields,
            JoinType::Outer,
        )
        .collect()?
        .fill_null(FillNullStrategy::Zero)?)
}

pub fn get_desired_instances(path: &PathBuf) -> Result<LazyFrame> {
    if let Ok(reader) = CsvReader::from_path(path) {
        Ok(reader.has_header(true).finish()?.lazy())
    } else {
        warn!(
            "Provided instances file: {:?} not found, using all instances",
            path
        );
        Err(anyhow::Error::msg("No instances file"))
    }
}
