use anyhow::Result;
use std::env;
use std::fs;

use portfolio_solver::csv_parser;
use portfolio_solver::datastructures::*;
use portfolio_solver::portfolio_simulator;
use portfolio_solver::solver;

fn main() -> Result<()> {
    let config_path = env::args().nth(1).expect("No json config provided!");
    let config_str = fs::read_to_string(config_path)
        .expect("Provided config file does not exist");
    let config: Config = serde_json::from_str(&config_str)
        .expect("Error while reading config file");
    let k = config.num_cores;
    let num_seeds = config.num_seeds;
    let out_file = config.out_file.clone();
    let data = csv_parser::Data::new(config);
    let df_config = DataframeConfig::new();
    println!("{data}");
    let portfolio = solver::solve(&data, k as usize);
    println!("{portfolio}");
    let portfolio_runs = portfolio_simulator::simulate_portfolio_execution(
        &data,
        portfolio,
        num_seeds,
        &df_config.instance_fields,
    );
    csv_parser::df_to_csv_for_performance_profiles(
        &portfolio_runs,
        &df_config,
        &out_file,
    );
    Ok(())
}
