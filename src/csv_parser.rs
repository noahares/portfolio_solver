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

pub struct DataframeConfig<'a> {
    schema: Schema,
    in_fields: Vec<String>,
    out_fields: Vec<&'a str>,
    pub instance_fields: Vec<&'a str>,
    sort_order: Vec<&'a str>,
}

impl DataframeConfig<'_> {
    pub fn new() -> Self {
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
        let instance_fields = vec!["instance", "k"];
        let sort_order = {
            let mut sort_order = instance_fields.clone();
            sort_order.extend(vec!["algorithm"]);
            sort_order
        };
        Self {
            schema,
            in_fields: kahypar_columns,
            out_fields: target_columns,
            instance_fields,
            sort_order,
        }
    }
}

impl Default for DataframeConfig<'_> {
    fn default() -> Self {
        Self::new()
    }
}

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
    pub fn new(config: Config) -> Result<Self> {
        let Config {
            files: paths,
            quality_lb: quality_lb_path,
            num_cores: k,
        } = config;
        let df_config = DataframeConfig::new();
        let df = preprocess_df(paths.as_ref(), &df_config)?.collect()?;

        let instances = extract_string_col(&df, "instance")?;
        let algorithms = extract_string_col(&df, "algorithm")?;
        let num_instances = df_config
            .instance_fields
            .iter()
            .map(|field| {
                Ok::<usize, PolarsError>(df.column(field)?.unique()?.len())
            })
            .fold_ok(1, Mul::mul)?;
        let num_algorithms = algorithms.len();
        let best_per_instance = best_per_instance(
            df.clone().lazy(),
            &df_config.instance_fields,
            "quality",
        )
        .collect()?
        .column("best_quality")?
        .f64()?
        .to_ndarray()?
        .to_owned();
        let quality_lb = read_quality_lb(quality_lb_path)?;
        assert!(best_per_instance.iter().all(|val| val.abs() >= EPSILON));
        let stats = stats(
            df.clone().lazy(),
            k,
            &df_config.instance_fields,
            Shape::from(ndarray::Dim([
                num_instances,
                num_algorithms,
                k as usize,
            ])),
            quality_lb.lazy(),
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

pub fn preprocess_df<T: AsRef<str>>(
    paths: &[T],
    config: &DataframeConfig,
) -> Result<LazyFrame> {
    let read_df = |path: &T| -> Result<LazyFrame> {
        Ok(CsvReader::from_path(path.as_ref())
            .unwrap_or_else(|_| {
                panic!("No csv file found under {}", path.as_ref())
            })
            .with_comment_char(Some(b'#'))
            .has_header(true)
            .with_columns(Some(config.in_fields.clone()))
            .with_dtypes(Some(&config.schema))
            .finish()?
            .lazy()
            .rename(config.in_fields.iter(), config.out_fields.iter())
            .with_columns([
                col("instance").apply(
                    |s: Series| {
                        Ok(s.utf8()?
                            .into_no_null_iter()
                            .map(fix_instance_names)
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
        config
            .sort_order
            .iter()
            .map(|o| col(o))
            .collect::<Vec<Expr>>(),
        vec![false; 3],
        false,
    ))
}

fn read_quality_lb(path: String) -> Result<DataFrame> {
    Ok(CsvReader::from_path(&path)
       .unwrap_or_else(|_| {
           panic!("No csv file found under {}, generate it with the dedicated binary", path)
       })
       .with_comment_char(Some(b'#'))
       .has_header(true)
       .with_columns(Some(vec!["instance".to_string(), "k".into(), "quality_lb".into()]))
       .finish()?)
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

fn extract_string_col(
    df: &DataFrame,
    column_name: &str,
) -> Result<ndarray::Array1<String>> {
    Ok(ndarray::Array1::from_iter(
        df.column(column_name)
            .unwrap_or_else(|_| {
                panic!("No column {} in dataframe", column_name)
            })
            .unique()?
            .utf8()?
            .into_no_null_iter()
            .map(|s| s.to_string())
            .sorted(),
    ))
}

pub fn best_per_instance(
    df: LazyFrame,
    instance_fields: &[&str],
    target_field: &str,
) -> LazyFrame {
    df.groupby_stable(instance_fields)
        .agg([min(target_field).prefix("best_")])
}

fn stats(
    df: LazyFrame,
    sample_size: u32,
    instance_fields: &[&str],
    shape: Shape<ndarray::Dim<[usize; 3]>>,
    quality_lb: LazyFrame,
) -> Result<ndarray::Array3<f64>> {
    let possible_repeats = df! {
        "sample_size" => Vec::from_iter(1..=sample_size)
    }?;
    // let fastest_per_instance_df = best_per_instance(df.clone(), instance_fields, "time");

    let stats =
        df.groupby_stable([&["algorithm"], instance_fields].concat())
            .agg([
                mean("quality").prefix("mean_"),
                col("quality").std(1).prefix("std_"),
            ])
            .join(
                quality_lb,
                instance_fields.iter().map(|field| col(field)).collect_vec(),
                instance_fields.iter().map(|field| col(field)).collect_vec(),
                JoinType::Inner,
            )
            .cross_join(possible_repeats.lazy())
            .select([
                col("mean_quality"),
                col("std_quality"),
                col("sample_size"),
                col("quality_lb"),
                as_struct(&[
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
                .alias("e_min"),
            ])
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

    use crate::{csv_parser::Data, datastructures::Config};

    #[test]
    fn test_dataframe() {
        let config = Config {
            files: vec![
                "data/test/algo1.csv".to_string(),
                "data/test/algo2.csv".into(),
            ],
            quality_lb: "data/test/quality_lb.csv".to_string(),
            num_cores: 2,
        };
        let data = Data::new(config).expect("Error while reading data");
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
            quality_lb: "data/test/quality_lb.csv".to_string(),
            num_cores: 2,
        };
        let data = Data::new(config).expect("Error while reading data");
        assert_eq!(data.num_instances, 4);
        assert_eq!(data.num_algorithms, 2);
        assert_eq!(data.best_per_instance, arr1(&[1.0, 7.0, 22.0, 1.0]));
    }
}
