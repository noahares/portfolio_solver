use itertools::{izip, Itertools};
use ndarray::Shape;
use polars::{
    lazy::dsl::{as_struct, Expr, GetOutput},
    prelude::*,
};
use std::ops::Mul;

use anyhow::Result;

pub struct Data {
    pub df: DataFrame,
    pub instances: ndarray::Array1<String>,
    pub algorithms: ndarray::Array1<String>,
    pub best_per_instance: ndarray::Array1<f64>,
    pub stats: ndarray::Array3<f64>,
    pub num_instances: usize,
    pub num_algorithms: usize,
}

impl Data {
    pub fn new(path: &str, k: u32) -> Result<Self> {
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
        let df = CsvReader::from_path(path)?
            .with_comment_char(Some(b'#'))
            .has_header(true)
            .with_columns(Some(kahypar_columns.clone()))
            .with_dtypes(Some(&schema))
            .finish()?
            .lazy()
            .rename(kahypar_columns, target_columns)
            .sort_by_exprs(
                sort_order.iter().map(|o| col(o)).collect::<Vec<Expr>>(),
                vec![false; 3],
                false,
            )
            .collect()?;

        let instances = ndarray::Array1::from_iter(
            df.column("instance")?
                .unique()?
                .utf8()?
                .into_no_null_iter()
                .map(|s| s.to_string()),
        );
        let algorithms = ndarray::Array1::from_iter(
            df.column("algorithm")?
                .unique()?
                .utf8()?
                .into_no_null_iter()
                .map(|s| s.to_string()),
        );
        let instance_multiplier = instance_fields
            .iter()
            .map(|field| {
                Ok::<usize, PolarsError>(df.column(field)?.unique()?.len())
            })
            .fold_ok(1, Mul::mul)?;
        let num_instances = instances.len() * instance_multiplier;
        let num_algorithms = algorithms.len();
        let best_per_instance =
            best_per_instance(df.clone().lazy(), &instance_fields)?;
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

fn best_per_instance(
    df: LazyFrame,
    instance_fields: &[&str],
) -> Result<ndarray::Array1<f64>> {
    let best_per_instance = df
        .groupby([&["instance"], instance_fields].concat())
        .agg([min("quality").alias("best")])
        .collect()?;
    Ok(best_per_instance
        .column("best")?
        .f64()?
        .to_ndarray()?
        .to_owned())
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

    let stats = df
        .groupby([&["algorithm", "instance"], instance_fields].concat())
        .agg([
            mean("quality").alias("mean_quality"),
            col("quality").std(1).alias("std_quality"),
        ])
        .cross_join(possible_repeats.lazy())
        .select([
            col("algorithm"),
            col("instance"),
            col("sample_size"),
            as_struct(&[
                col("mean_quality"),
                col("std_quality"),
                col("sample_size"),
            ])
            .apply(
                |s| {
                    let data = s.struct_()?;
                    let (mean_series, std_series, sample_series) = (
                        &data.fields()[0].f64()?,
                        &data.fields()[1].f64()?,
                        &data.fields()[2].u32()?,
                    );
                    let result: Float64Chunked = izip!(
                        mean_series.into_iter(),
                        std_series.into_iter(),
                        sample_series.into_iter()
                    )
                    .map(|(opt_mean, opt_std, opt_s)| {
                        match (opt_mean, opt_std, opt_s) {
                            (Some(mean), Some(std), Some(s)) => {
                                Some(expected_normdist_min(mean, std, s))
                            }
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

fn expected_normdist_min(mean: f64, std: f64, sample_size: u32) -> f64 {
    mean - std * expected_maximum_approximation(sample_size)
}

fn expected_maximum_approximation(sample_size: u32) -> f64 {
    f64::sqrt(2.0 * f64::ln(sample_size as f64))
}
