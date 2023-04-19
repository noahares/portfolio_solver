use anyhow::Result;
use core::fmt;
use itertools::Itertools;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

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
            1..=num_single_threaded_algorithms.min(num_cores) as usize,
        );
        let core_distribution_sample = (0..num_algorithms_in_portfolio - 1)
            .map(|_| rng.gen_range(1..=num_cores))
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
    pub initial_portfolio: Option<Portfolio>,
    pub final_portfolio: Portfolio,
    pub gap: f64,
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
