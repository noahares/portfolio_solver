use std::path::PathBuf;

use itertools::{izip, Itertools};
use log::warn;
use polars::prelude::*;

use anyhow::{Context, Result};

use crate::datastructures::*;

pub fn fix_instance_names(instance: &str) -> String {
    if instance.ends_with("scotch") {
        instance.replace("scotch", "graph")
    } else {
        instance.to_string()
    }
}

pub fn extract_instance_columns(
    df: &DataFrame,
    instance_fields: &[&str],
) -> Result<ndarray::Array1<Instance>> {
    let unique_instances_df = df
        .clone()
        .lazy()
        .unique_stable(
            Some(instance_fields.iter().map(|s| s.to_string()).collect_vec()),
            UniqueKeepStrategy::First,
        )
        .collect()?;
    let instance_it = unique_instances_df
        .column("instance")?
        .utf8()?
        .into_no_null_iter();
    let k_it = unique_instances_df.column("k")?.i64()?.into_no_null_iter();
    let eps_it = unique_instances_df
        .column("feasibility_threshold")?
        .f64()?
        .into_no_null_iter();
    Ok(ndarray::Array1::from_iter(
        izip!(instance_it, k_it, eps_it)
            .map(|(i, k, e)| Instance::new(i.to_string(), k as u32, e)),
    ))
}

pub fn extract_algorithm_columns(
    df: &DataFrame,
    algorithm_fields: &[&str],
) -> Result<ndarray::Array1<Algorithm>> {
    let unique_algorithm_df = df
        .clone()
        .lazy()
        .unique_stable(
            Some(algorithm_fields.iter().map(|s| s.to_string()).collect_vec()),
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

pub fn best_per_instance_time(
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

pub fn column_to_f64_array(
    df: &DataFrame,
    column_name: &str,
) -> Result<ndarray::Array1<f64>> {
    Ok(df.column(column_name)?.f64()?.to_ndarray()?.to_owned())
}

pub fn stats_by_sampling(
    df: LazyFrame,
    sample_size: u32,
    instance_fields: &[&str],
    algorithm_fields: &[&str],
) -> Result<LazyFrame> {
    let columns = [instance_fields, algorithm_fields]
        .concat()
        .iter()
        .map(|f| col(f))
        .collect_vec();

    let sort_order = [
        DataframeConfig::global().sort_order.clone(),
        ["sample_size"].to_vec(),
    ]
    .concat();
    let sort_exprs = sort_order.iter().map(|o| col(o)).collect::<Vec<Expr>>();
    let sort_options =
        vec![false; instance_fields.len() + algorithm_fields.len() + 1];
    let samples_per_repeats: Vec<LazyFrame> = (1_u64..=sample_size as u64)
        .map(|s| {
            df.clone()
                .groupby(columns.clone())
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

pub fn cleanup_missing_rows(
    df: DataFrame,
    k: u32,
    instance_fields: &[&str],
    algorithm_fields: &[&str],
) -> Result<DataFrame> {
    let algorithm_series = df
        .select(algorithm_fields)?
        .unique_stable(None, UniqueKeepStrategy::First)?
        .lazy();
    let instance_series = df
        .select(instance_fields)?
        .unique_stable(None, UniqueKeepStrategy::First)?
        .lazy();
    let possible_repeats = df! {
        "sample_size" => Vec::from_iter(1..=k)
    }
    .unwrap();
    let full_df = instance_series
        .cross_join(algorithm_series)
        .cross_join(possible_repeats.lazy())
        .collect()?;
    let target_columns =
        [instance_fields, algorithm_fields, &["sample_size"]].concat();
    Ok(df
        .outer_join(&full_df, &target_columns, &target_columns)?
        .fill_null(FillNullStrategy::MaxBound)?)
}

pub fn filter_algorithms_by_slowdown(
    df: LazyFrame,
    instance_fields: &[&str],
    algorithm_fields: &[&str],
    slowdown_ratio: f64,
) -> Result<LazyFrame> {
    let gmean = |s: Series| -> Result<Series, PolarsError> {
        let gmean = s.f64()?.into_no_null_iter().map(|v| v.ln()).sum::<f64>()
            / s.len() as f64;
        Ok(Series::new("gmean", &[gmean]))
    };
    let best_per_instance_time_df =
        best_per_instance_time(df.clone(), instance_fields, "quality");
    let gmean_best_per_instance = best_per_instance_time_df
        .select([col("best_time")
            .apply(gmean, GetOutput::from_type(DataType::Float64))
            .alias("gmean")])
        .collect()?
        .column("gmean")?
        .f64()?
        .into_no_null_iter()
        .last()
        .context("empty dataframe")?;
    let filtered_df = df
        .clone()
        .groupby(algorithm_fields)
        .agg([col("time")
            .apply(gmean, GetOutput::from_type(DataType::Float64))
            .first()
            .alias("gmean")])
        .filter(
            col("gmean").lt(lit(slowdown_ratio * gmean_best_per_instance)),
        );
    Ok(df.join(
        filtered_df,
        algorithm_fields
            .iter()
            .map(|field| col(field))
            .collect_vec(),
        algorithm_fields
            .iter()
            .map(|field| col(field))
            .collect_vec(),
        JoinType::Inner,
    ))
}

pub fn best_per_instance_count(
    df: DataFrame,
    instance_fields: &[&str],
    algorithm_fields: &[&str],
    target_field: &str,
) -> Result<DataFrame> {
    let algorithm_series = df
        .select(algorithm_fields)?
        .unique_stable(None, UniqueKeepStrategy::First)?;
    Ok(df
        .lazy()
        .groupby_stable(instance_fields)
        .agg([col("*")
            .sort_by(vec![col(target_field)], vec![false])
            .first()])
        .select(
            algorithm_fields
                .iter()
                .map(|field| col(field))
                .collect_vec(),
        )
        .groupby_stable(algorithm_fields)
        .agg([col("*"), count().alias("count").cast(DataType::Float64)])
        .collect()?
        .outer_join(&algorithm_series, algorithm_fields, algorithm_fields)?
        .fill_null(FillNullStrategy::Zero)?)
}

pub fn filter_desired_instances(
    df: LazyFrame,
    graphs_path: &PathBuf,
    num_parts: &Vec<i64>,
    feasibility_thresholds: &Vec<f64>,
    instance_fields: &[&str],
) -> Result<LazyFrame> {
    if let Ok(reader) = CsvReader::from_path(graphs_path) {
        let graph_df = reader
            .has_header(true)
            .finish()?
            .lazy()
            .rename(["graph"], ["instance"]);
        let k_df = df! {
            "k" => num_parts
        }?;
        let eps_df = df! {
            "feasibility_threshold" => feasibility_thresholds
        }?;
        Ok(df.join(
            graph_df.cross_join(k_df.lazy()).cross_join(eps_df.lazy()),
            instance_fields.iter().map(|field| col(field)).collect_vec(),
            instance_fields.iter().map(|field| col(field)).collect_vec(),
            JoinType::Inner,
        ))
    } else {
        warn!(
            "Provided graph file: {:?} not found, using all graphs",
            graphs_path
        );
        Ok(df)
    }
}
