use anyhow::Result;
use portfolio_solver::{csv_parser, datastructures::*, portfolio_simulator};
use std::{env, fs};

fn main() -> Result<()> {
    let config_path = env::args().nth(1).expect("No json config provided!");
    let config_str = fs::read_to_string(config_path)
        .expect("Provided config file does not exist");
    let PortfolioExecutorConfig {
        files,
        portfolio,
        num_seeds,
        num_cores,
        out,
    } = serde_json::from_str(&config_str)
        .expect("Error while reading config file");

    let df_config = DataframeConfig::new();
    let df = csv_parser::preprocess_df(&files, &df_config).collect()?;
    let algorithms = csv_parser::extract_algorithm_columns(
        &df,
        &df_config.algorithm_fields,
    );
    let simulation = portfolio_simulator::simulation_df(
        &df,
        &algorithms,
        &portfolio,
        num_seeds,
        &df_config.instance_fields,
        &df_config.algorithm_fields,
        num_cores,
    );
    csv_parser::df_to_csv_for_performance_profiles(
        simulation, &df_config, &out,
    );
    Ok(())
}
