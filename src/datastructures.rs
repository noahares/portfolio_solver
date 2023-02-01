use core::fmt;
use serde::{Deserialize, Serialize};

pub type Instance = String;
pub type Algorithm = String;

#[derive(Serialize, Deserialize)]
pub struct Config {
    pub files: Vec<String>,
    pub quality_lb: String,
    pub num_cores: u32,
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
