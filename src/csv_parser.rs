use core::fmt;
use itertools::{izip, Itertools};
use ndarray::Shape;
use polars::{
    lazy::dsl::{as_struct, Expr, GetOutput},
    prelude::*,
    series::IsSorted,
};
use std::f64::EPSILON;
use std::ops::Mul;

use anyhow::Result;

use crate::datastructures::*;

pub struct Data {
    pub df: DataFrame,
    pub instances: ndarray::Array1<Instance>,
    pub algorithms: ndarray::Array1<Algorithm>,
    pub best_per_instance: ndarray::Array1<f64>,
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
    pub fn new<T: AsRef<str>>(paths: &[T], k: u32) -> Result<Self> {
        let schema = Schema::from(
            vec![Field::new("km1", DataType::Float64)].into_iter(),
        );
        let kahypar_columns = vec![
            "algorithm".to_string(),
            "graph".into(),
            "k".into(),
            "imbalance".into(),
            "km1".into(),
            "totalPartitionTime".into(),
        ];
        let target_columns = vec![
            "algorithm",
            "instance",
            "k",
            "feasibility_score",
            "quality",
            "time",
        ];
        let instance_fields = vec!["k"];
        let sort_order = {
            let mut sort_order = vec!["instance"];
            sort_order.extend(instance_fields.iter());
            sort_order.extend(vec!["algorithm"].iter());
            sort_order
        };
        let df = preprocess_df(
            paths,
            schema,
            kahypar_columns,
            target_columns,
            sort_order,
        )?
        .collect()?;

        let instances = extract_string_col(&df, "instance")?;
        let algorithms = extract_string_col(&df, "algorithm")?;
        let instance_multiplier = instance_fields
            .iter()
            .map(|field| {
                Ok::<usize, PolarsError>(df.column(field)?.unique()?.len())
            })
            .fold_ok(1, Mul::mul)?;
        let num_instances = instances.len() * instance_multiplier;
        let num_algorithms = algorithms.len();
        let best_per_instance =
            best_per_instance(df.clone().lazy(), &instance_fields)
                .collect()?
                .column("best")?
                .f64()?
                .to_ndarray()?
                .to_owned();
        let stats = stats(
            df.clone().lazy(),
            k,
            &instance_fields,
            Shape::from(ndarray::Dim([
                num_instances,
                num_algorithms,
                k as usize,
            ])),
        )?;
        assert_eq!(df.column("instance")?.is_sorted(), IsSorted::Ascending);
        Ok(Self {
            df,
            instances,
            algorithms,
            best_per_instance,
            stats,
            num_instances,
            num_algorithms,
        })
    }
}

fn preprocess_df<T: AsRef<str>>(
    paths: &[T],
    schema: Schema,
    in_fields: Vec<String>,
    out_fields: Vec<&str>,
    sort_order: Vec<&str>,
) -> Result<LazyFrame> {
    let read_df = |path: &T| -> Result<LazyFrame> {
        Ok(CsvReader::from_path(path.as_ref())
            .expect("No csv file found under {path}")
            .with_comment_char(Some(b'#'))
            .has_header(true)
            .with_columns(Some(in_fields.clone()))
            .with_dtypes(Some(&schema))
            .finish()?
            .lazy()
            .rename(in_fields.iter(), out_fields.iter())
            .with_columns([
                col("instance").apply(
                    |s: Series| {
                        Ok(s.utf8()?
                            .into_no_null_iter()
                            .map(|str| str.replace("scotch", "graph"))
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

    let dataframes: Vec<LazyFrame> =
        paths.iter().map(|path| read_df(path).unwrap()).collect();
    Ok(concat(dataframes, false, false)?.sort_by_exprs(
        sort_order.iter().map(|o| col(o)).collect::<Vec<Expr>>(),
        vec![false; 3],
        false,
    ))
}

fn extract_string_col(
    df: &DataFrame,
    column_name: &str,
) -> Result<ndarray::Array1<String>> {
    Ok(ndarray::Array1::from_iter(
        df.column(column_name)
            .expect("{column_name} not found in data frame")
            .unique()?
            .utf8()?
            .into_no_null_iter()
            .map(|s| s.to_string())
            .sorted(),
    ))
}

fn best_per_instance(df: LazyFrame, instance_fields: &[&str]) -> LazyFrame {
    df.groupby_stable([&["instance"], instance_fields].concat())
        .agg([min("quality").alias("best")])
}

fn stats(
    df: LazyFrame,
    sample_size: u32,
    instance_fields: &[&str],
    shape: Shape<ndarray::Dim<[usize; 3]>>,
) -> Result<ndarray::Array3<f64>> {
    let possible_repeats = df! {
        "sample_size" => Vec::from_iter(1..=sample_size)
    }?;
    let best_per_instance = best_per_instance(df.clone(), instance_fields);

    let stats = df
        .groupby_stable([&["algorithm", "instance"], instance_fields].concat())
        .agg([
            mean("quality").alias("mean_quality"),
            col("quality").std(1).alias("std_quality"),
        ])
        .join(
            best_per_instance,
            [col("instance"), col("k")],
            [col("instance"), col("k")],
            JoinType::Inner,
        )
        .cross_join(possible_repeats.lazy())
        .select(
            [
                col("mean_quality"),
                col("std_quality"),
                col("sample_size"),
                col("best"),
                as_struct(&[
                    col("mean_quality"),
                    col("std_quality"),
                    col("sample_size"),
                    col("best"),
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
                .alias("e_min"),
            ],
        )
        .collect()?;
    let stats_array: ndarray::Array3<f64> =
        ndarray::Array3::<f64>::from_shape_vec(
            shape,
            stats
                .column("e_min")?
                .f64()?
                .into_no_null_iter()
                .collect::<Vec<f64>>(),
        )?;
    Ok(stats_array)
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

#[cfg(test)]
mod tests {
    use ndarray::{arr1, aview2, Axis};

    use crate::csv_parser::Data;

    #[test]
    fn test_dataframe() {
        let csv_paths = vec!["data/test/algo1.csv", "data/test/algo2.csv"];
        let num_cores = 2;
        let data = Data::new(&csv_paths, num_cores)
            .expect("Error while reading data");
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
        let csv_paths = vec!["data/test/algo2.csv", "data/test/algo3.csv"];
        let num_cores = 2;
        let data = Data::new(&csv_paths, num_cores)
            .expect("Error while reading data");
        assert_eq!(data.num_instances, 4);
        assert_eq!(data.num_algorithms, 2);
        assert_eq!(data.best_per_instance, arr1(&[1.0, 7.0, 22.0, 1.0]));
    }
}
