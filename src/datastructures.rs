use serde::{Deserialize, Serialize};

pub type Instance = String;
pub type Algorithm = String;

#[derive(Serialize, Deserialize)]
pub struct Config {
    pub files: Vec<String>,
    pub num_cores: u32,
}
