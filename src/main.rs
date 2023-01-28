use anyhow::Result;
use std::env;
use std::fs;

use portfolio_solver::csv_parser::Data;
use portfolio_solver::datastructures::*;
use portfolio_solver::portfolio_simulator;
use portfolio_solver::solver;

fn main() -> Result<()> {
    let config_path = env::args().nth(1).expect("No json config provided!");
    let config_str = fs::read_to_string(config_path)
        .expect("Provided config file does not exist");
    let config: Config = serde_json::from_str(&config_str)?;
    let data = Data::new(&config.files, config.num_cores)?;
    let result = solver::solve(&data, config.num_cores as usize)?;
    dbg!("assignment {:?}", result);
    dbg!("{:?}", data.algorithms);
    Ok(())
}
