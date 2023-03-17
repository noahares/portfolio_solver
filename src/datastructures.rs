use anyhow::Result;
use core::fmt;
use itertools::Itertools;
use polars::datatypes::*;
use polars::prelude::Schema;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, PartialOrd, Clone)]
pub struct Instance {
    pub graph: String,
    pub k: u32,
    pub feasibility_threshold: f64,
}

impl Instance {
    pub fn new(graph: String, k: u32, feasibility_threshold: f64) -> Self {
        Self {
            graph,
            k,
            feasibility_threshold,
        }
    }
}

#[derive(
    Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize, PartialOrd,
)]
pub struct Algorithm {
    pub algorithm: String,
    pub num_threads: u32,
}

impl Algorithm {
    pub fn new(algorithm: String, num_threads: u32) -> Self {
        Self {
            algorithm,
            num_threads,
        }
    }
}

impl fmt::Display for Algorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.algorithm, self.num_threads)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Timeout(pub f64);
impl Default for Timeout {
    fn default() -> Self {
        Timeout(900.0)
    }
}

impl FromStr for Timeout {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(Self(s.parse::<f64>()?))
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Config {
    pub files: Vec<String>,
    #[serde(default)]
    pub graphs: String,
    #[serde(default = "default_ks")]
    pub ks: Vec<i64>,
    #[serde(default = "default_feasibility_thresholds")]
    pub feasibility_thresholds: Vec<f64>,
    pub num_cores: u32,
    pub slowdown_ratio: f64,
    pub num_seeds: u32,
    pub out_dir: String,
    #[serde(default)]
    pub timeout: Timeout,
}

fn default_ks() -> Vec<i64> {
    vec![2, 4, 8, 16, 32, 64, 128]
}

fn default_feasibility_thresholds() -> Vec<f64> {
    vec![0.03]
}

#[derive(Serialize, Deserialize)]
pub struct QualityLowerBoundConfig {
    pub files: Vec<String>,
    pub out: String,
}

#[derive(Serialize, Deserialize)]
pub struct PortfolioExecutorConfig {
    pub files: Vec<String>,
    pub portfolios: Vec<Portfolio>,
    pub num_seeds: u32,
    pub num_cores: u32,
    pub out: String,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct Portfolio {
    pub name: String,
    pub resource_assignments: Vec<(Algorithm, f64)>,
}

impl fmt::Display for Portfolio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (algo, cores) in &self.resource_assignments {
            writeln!(f, "{}: {}", algo, cores)?;
        }
        Ok(())
    }
}

impl Portfolio {
    pub fn random<'a, I>(algorithms: I, num_cores: u32, seed: u64) -> Self
    where
        I: IntoIterator<Item = &'a Algorithm>,
    {
        let single_threaded_algorithms = algorithms
            .into_iter()
            .filter(|&a| a.num_threads == 1)
            .collect_vec();
        let num_single_threaded_algorithms =
            single_threaded_algorithms.len() as u32;
        if num_single_threaded_algorithms == 0 {
            return Self {
                name: String::from("random_portfolio"),
                resource_assignments: Vec::new(),
            };
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let num_algorithms_in_portfolio = rng.gen_range(
            1..num_single_threaded_algorithms.min(num_cores) as usize,
        );
        let core_distribution_sample = (0..num_algorithms_in_portfolio - 1)
            .map(|_| rng.gen_range(1..num_cores))
            .sorted()
            .collect_vec();
        let cores_per_algorithm =
            [core_distribution_sample.clone(), vec![num_cores]]
                .concat()
                .iter()
                .zip([vec![0], core_distribution_sample].concat().iter())
                .map(|(a, b)| a - b)
                .collect_vec();
        let mut random_algorithms = single_threaded_algorithms;
        let random_algorithms = random_algorithms
            .partial_shuffle(&mut rng, num_algorithms_in_portfolio)
            .0;
        Self {
            name: String::from("random_portfolio"),
            resource_assignments: random_algorithms
                .iter()
                .zip(cores_per_algorithm.iter())
                .map(|(&a, v)| (a.clone(), *v as f64))
                .collect_vec(),
        }
    }
}

pub struct OptimizationResult {
    pub initial_portfolio: Portfolio,
    pub final_portfolio: Portfolio,
    pub gap: f64,
}

#[derive(Debug)]
pub struct DataframeConfig<'a> {
    pub schema: Schema,
    pub in_fields: Vec<String>,
    pub out_fields: Vec<&'a str>,
    pub instance_fields: Vec<&'a str>,
    pub algorithm_fields: Vec<&'a str>,
    pub sort_order: Vec<&'a str>,
}

impl DataframeConfig<'_> {
    pub fn new() -> Self {
        let schema = Schema::from(
            vec![Field::new("km1", DataType::Float64)].into_iter(),
        );
        let kahypar_columns = vec![
            "algorithm".to_string(),
            "num_threads".into(),
            "graph".into(),
            "k".into(),
            "epsilon".into(),
            "imbalance".into(),
            "km1".into(),
            "totalPartitionTime".into(),
            "failed".into(),
            "timeout".into(),
        ];
        let target_columns = vec![
            "algorithm",
            "num_threads",
            "instance",
            "k",
            "feasibility_threshold",
            "feasibility_score",
            "quality",
            "time",
            "failed",
            "timeout",
        ];
        let instance_fields = vec!["instance", "k", "feasibility_threshold"];
        let algorithm_fields = vec!["algorithm", "num_threads"];
        let sort_order = {
            let mut sort_order = instance_fields.clone();
            sort_order.extend(&algorithm_fields);
            sort_order
        };
        Self {
            schema,
            in_fields: kahypar_columns,
            out_fields: target_columns,
            instance_fields,
            algorithm_fields,
            sort_order,
        }
    }
}

impl Default for DataframeConfig<'_> {
    fn default() -> Self {
        Self::new()
    }
}

use clap::Parser;
use std::{path::PathBuf, str::FromStr};

#[derive(Parser)]
#[command(author, version, about)]
pub struct Args {
    #[arg(short, long)]
    pub config: PathBuf,
    #[arg(short, long)]
    pub slowdown_ratio: Option<f64>,
    #[arg(short, long)]
    pub out_dir: Option<PathBuf>,
    #[arg(short, long, value_parser)]
    pub timeout: Option<Timeout>,
    #[arg(short = 'k', long)]
    pub num_cores: Option<u32>,
    #[arg(short, long)]
    pub initial_portfolio: bool,
    #[arg(short, long)]
    pub random_portfolio: bool,
}

#[cfg(test)]
mod tests {
    use super::Algorithm;

    use super::Portfolio;

    #[test]
    fn test_random_portfolio() {
        let algorithms = vec![
            Algorithm::new("algo1".into(), 1),
            Algorithm::new("algo2".into(), 1),
            Algorithm::new("algo3".into(), 1),
            Algorithm::new("algo4".into(), 1),
            Algorithm::new("algo5".into(), 2),
        ];
        for seed in 0..9 {
            let result = Portfolio::random(&algorithms, 16, seed);
            assert_eq!(
                result
                    .resource_assignments
                    .iter()
                    .map(|(_, c)| c)
                    .sum::<f64>(),
                16.0
            );
        }
    }
}
