use core::fmt;
use polars::datatypes::*;
use polars::prelude::Schema;
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub struct Instance {
    pub graph: String,
    pub k: u32,
}

impl Instance {
    pub fn new(graph: String, k: u32) -> Self {
        Self { graph, k }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
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

#[derive(Serialize, Deserialize)]
pub struct Config {
    pub files: Vec<String>,
    pub quality_lb: String,
    pub num_cores: u32,
    pub slowdown_ratio: f64,
}

#[derive(Serialize, Deserialize)]
pub struct QualityLowerBoundConfig {
    pub files: Vec<String>,
    pub out: String,
}

#[derive(Debug, PartialEq)]
pub struct SolverResult {
    pub resource_assignments: Vec<(Algorithm, f64)>,
}

impl fmt::Display for SolverResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (algo, cores) in &self.resource_assignments {
            writeln!(f, "{}: {}", algo, cores)?;
        }
        Ok(())
    }
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
            "feasibility_score",
            "quality",
            "time",
            "failed",
            "timeout",
        ];
        let instance_fields = vec!["instance", "k"];
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
