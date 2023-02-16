use itertools::{izip, Itertools};
use polars::{
    lazy::dsl::{as_struct, GetOutput},
    prelude::*,
};

use anyhow::Result;

use crate::datastructures::*;

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

pub fn stats(
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
                // NOTE: this filter does probably nothing atm
                .filter(col("quality").lt(lit(std::f64::MAX)))
                .mean()
                .fill_null(lit(std::f64::MAX))
                .prefix("mean_"),
            col("quality")
                // NOTE: this filter does probably nothing atm
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

pub fn filter_slowdown(
    df: LazyFrame,
    instance_fields: &[&str],
    slowdown_ratio: f64,
    best_per_instance_time_df: LazyFrame,
    slowdown_penalty: f64,
) -> LazyFrame {
    df.join(
        best_per_instance_time_df,
        instance_fields.iter().map(|field| col(field)).collect_vec(),
        instance_fields.iter().map(|field| col(field)).collect_vec(),
        JoinType::Inner,
    )
    .with_column(
        when(col("time").lt(col("best_time") * lit(slowdown_ratio)))
            .then(col("quality"))
            .otherwise(slowdown_penalty)
            .alias("quality"),
    )
    // .filter(col("time").lt(col("best_time") * lit(slowdown_ratio)))
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
        .agg([count().alias("count").cast(DataType::Float64)])
        .collect()
        .expect("Error counting best per instance")
        .outer_join(&algorithm_series, algorithm_fields, algorithm_fields)
        .expect("Error filling best per instance count")
        .fill_null(FillNullStrategy::Zero)
        .expect("Error filling best per instance count")
}
