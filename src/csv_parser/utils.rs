use itertools::{izip, Itertools};
use polars::prelude::*;

use anyhow::Result;

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
    let eps_it = unique_instances_df
        .column("feasibility_threshold")
        .expect("No field `feasibility_threshold`")
        .f64()
        .expect("Field `feasibility_threshold` should be a float")
        .into_no_null_iter();
    ndarray::Array1::from_iter(
        izip!(instance_it, k_it, eps_it)
            .map(|(i, k, e)| Instance::new(i.to_string(), k as u32, e)),
    )
}

pub fn extract_algorithm_columns(
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

pub fn stats_by_sampling(
    df: LazyFrame,
    sample_size: u32,
    instance_fields: &[&str],
    algorithm_fields: &[&str],
) -> LazyFrame {
    let columns = [instance_fields, algorithm_fields]
        .concat()
        .iter()
        .map(|f| col(f))
        .collect_vec();

    let sort_order =
        [DataframeConfig::new().sort_order, ["sample_size"].to_vec()].concat();
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
    concat(samples_per_repeats, false, false)
        .expect("Failed to build E_min dataframe")
        .sort_by_exprs(&sort_exprs, sort_options, false)
}

pub fn cleanup_missing_rows(
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

pub fn filter_algorithms_by_slowdown(
    df: LazyFrame,
    instance_fields: &[&str],
    algorithm_fields: &[&str],
    slowdown_ratio: f64,
) -> LazyFrame {
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
        .collect()
        .expect("Failed to get gmean value")["gmean"]
        .f64()
        .unwrap()
        .into_no_null_iter()
        .last()
        .unwrap();
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
    df.join(
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
    )
}

pub fn best_per_instance_count(
    df: DataFrame,
    instance_fields: &[&str],
    algorithm_fields: &[&str],
    target_field: &str,
) -> DataFrame {
    let algorithm_series = df
        .select(algorithm_fields)
        .expect("Best per instance ranking failed due to `algorithm_fields`")
        .unique_stable(None, UniqueKeepStrategy::First)
        .expect("Best per instance ranking failed due to `algorithm_fields`");
    df.lazy()
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
        .collect()
        .expect("Error counting best per instance")
        .outer_join(&algorithm_series, algorithm_fields, algorithm_fields)
        .expect("Error filling best per instance count")
        .fill_null(FillNullStrategy::Zero)
        .expect("Error filling best per instance count")
}

pub fn filter_desired_instances(
    df: LazyFrame,
    graphs_path: &String,
    num_parts: &Vec<i64>,
    feasibility_thresholds: &Vec<f64>,
    instance_fields: &[&str],
) -> LazyFrame {
    if let Ok(reader) = CsvReader::from_path(graphs_path) {
        let graph_df = reader
            .has_header(true)
            .finish()
            .expect("Failed to read graphs file")
            .lazy()
            .rename(["graph"], ["instance"]);
        let k_df = df! {
            "k" => num_parts
        }
        .unwrap();
        let eps_df = df! {
            "feasibility_threshold" => feasibility_thresholds
        }
        .unwrap();
        df.join(
            graph_df.cross_join(k_df.lazy()).cross_join(eps_df.lazy()),
            instance_fields.iter().map(|field| col(field)).collect_vec(),
            instance_fields.iter().map(|field| col(field)).collect_vec(),
            JoinType::Inner,
        )
    } else {
        println!("Provided graph file not found, using all graphs");
        df
    }
}
