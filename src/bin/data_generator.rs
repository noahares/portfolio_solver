use std::ops::Range;

use clap::Parser;
use polars::prelude::*;
use std::{fs, path::PathBuf};

use anyhow::Result;
use portfolio_solver::csv_parser;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
struct InstanceRangeConfig {
    mean: f64,
    std: f64,
    range: Range<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct AlgorithmConfig {
    instance_range_configs: Vec<InstanceRangeConfig>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct DataGeneratorConfig {
    algorithm_configs: Vec<AlgorithmConfig>,
    num_instances: usize,
    runs_per_instance: usize,
    seed: u64,
    out_path: PathBuf,
}

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    /// Path to the json config
    #[arg(short, long)]
    pub config: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config: DataGeneratorConfig =
        serde_json::from_str(&fs::read_to_string(args.config)?)?;
    let out_path = config.out_path.clone();
    let dataframe = generate_data(config)?;
    csv_parser::df_to_normalized_csv(dataframe, out_path)?;
    Ok(())
}

fn generate_data(config: DataGeneratorConfig) -> Result<LazyFrame> {
    let seed = config.seed;
    let runs_per_instance = config.runs_per_instance;
    let algorithm_dataframes = config.algorithm_configs
        .iter()
        .enumerate()
        .map(|(algo_idx, AlgorithmConfig { instance_range_configs })| -> Result<Vec<LazyFrame>> {
       Ok(instance_range_configs
           .iter()
           .map(move |InstanceRangeConfig {mean, std, range}| -> Result<Vec<LazyFrame>> {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let distrib = Normal::new(*mean, (*mean * *std).abs())?;
            Ok(range.clone()
                .map(|i| -> Result<LazyFrame> {
                let samples: Vec<f64> = distrib.sample_iter(&mut rng).take(runs_per_instance).collect();
                Ok(df! {
                    "algorithm" => vec![format!("{}{}", "algo", algo_idx); runs_per_instance],
                    "num_threads" => vec![1; runs_per_instance],
                    "instance" => vec![format!("{}{}", "graph", i); runs_per_instance],
                    "k" => vec![2; runs_per_instance],
                    "feasibility_threshold" => vec![0.0; runs_per_instance],
                    "feasibility_score" => vec![0.0; runs_per_instance],
                    "quality" => samples,
                    "time" => vec![1.0; runs_per_instance],
                    "failed" => vec![String::from("no"); runs_per_instance],
                    "timeout" => vec![String::from("no"); runs_per_instance],
                }?.lazy())
            })
            .filter_map(Result::ok)
            .collect())
       })
       .filter_map(Result::ok)
       .flatten()
       .collect())
    })
    .filter_map(Result::ok)
    .flatten()
    .collect::<Vec<LazyFrame>>();
    Ok(concat(algorithm_dataframes, false, false)?)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::{
        generate_data, AlgorithmConfig, DataGeneratorConfig,
        InstanceRangeConfig,
    };

    #[test]
    fn test_generate_data() {
        let config = DataGeneratorConfig {
            algorithm_configs: vec![
                AlgorithmConfig {
                    instance_range_configs: vec![
                        InstanceRangeConfig {
                            mean: 100.0,
                            std: 10.0,
                            range: (0..3),
                        },
                        InstanceRangeConfig {
                            mean: 50.0,
                            std: 10.0,
                            range: (3..5),
                        },
                    ],
                },
                AlgorithmConfig {
                    instance_range_configs: vec![
                        InstanceRangeConfig {
                            mean: 50.0,
                            std: 10.0,
                            range: (0..3),
                        },
                        InstanceRangeConfig {
                            mean: 100.0,
                            std: 10.0,
                            range: (3..5),
                        },
                    ],
                },
            ],
            seed: 42,
            num_instances: 5,
            runs_per_instance: 2,
            out_path: PathBuf::new(),
        };
        let data = generate_data(config).unwrap().collect().unwrap();
        assert_eq!(data.height(), 20);
    }
}
