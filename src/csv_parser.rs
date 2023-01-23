use itertools::izip;
use polars::{prelude::*, lazy::dsl::{as_struct, GetOutput}};

use anyhow::Result;

// pub fn read_csv(path: &str) -> Result<Dataframe> {
//     let mut datapoints: Vec<Datapoint> = Vec::new();
//     let mut algorithms = HashSet::new();
//     let mut instances = HashSet::new();
//     let mut rdr = csv::ReaderBuilder::new()
//         .comment(Some(b'#'))
//         .from_path(path)?;
//     for record in rdr.deserialize() {
//         let datapoint: Datapoint = record?;
//         algorithms.insert(datapoint.algorithm.clone());
//         instances.insert(datapoint.instance.clone());
//         datapoints.push(datapoint);
//     }
//     Ok(Dataframe { datapoints, algorithms: Vec::from_iter(algorithms), instances: Vec::from_iter(instances) })
// }

pub fn read_kahypar_csv(path: &str, k: u32) -> Result<()> {
    let kahypar_columns = vec!["algorithm".to_string(), "graph".into(), "k".into(), "imbalance".into(), "km1".into(), "partitionTime".into()];
    let df = CsvReader::from_path(path)?
        .with_comment_char(Some(b'#'))
        .has_header(true)
        .with_columns(Some(kahypar_columns))
        .finish()?;

    let best_per_instance = df.clone().lazy()
        .groupby(["graph", "k"])
        .agg([min("km1")])
        .collect()?;

    let possible_repeats = df!{
        "sample_size" => Vec::from_iter(1..k)
    }?;

    let stats = df.lazy()
        .groupby(["algorithm", "graph", "k"])
        .agg([
             mean("km1").alias("mean_km1"),
             col("km1").std(1).alias("std_km1"),
        ])
        .cross_join(possible_repeats.lazy())
        .select([
                col("algorithm"),
                col("graph"),
                col("sample_size"),
                as_struct(&[col("mean_km1"), col("std_km1"), col("sample_size")])
                .apply(
                    |s| {
                        let data = s.struct_()?;
                        let (mean_series, std_series, sample_series) = (&data.fields()[0].f64()?, &data.fields()[1].f64()?, &data.fields()[2].u32()?);
                        let result: Float64Chunked = izip!(mean_series.into_iter(), std_series.into_iter(), sample_series.into_iter())
                            .map(|(opt_mean, opt_std, opt_s)| match (opt_mean, opt_std, opt_s) {
                                (Some(mean), Some(std), Some(s)) => Some(expected_normdist_min(mean, std, s)),
                                _ => None
                            })
                        .collect();
                        Ok(result.into_series())
                    },
                    GetOutput::from_type(DataType::Float64)
                    ).alias("e_min")
        ])
        .collect()?;

    println!("{stats:#?}");
    Ok(())
}

fn expected_normdist_min(mean: f64, std: f64, sample_size: u32) -> f64 {
    mean - std * expected_maximum_approximation(sample_size)
}

fn expected_maximum_approximation(sample_size: u32) -> f64 {
    f64::sqrt(2.0 * f64::ln(sample_size as f64))
}
