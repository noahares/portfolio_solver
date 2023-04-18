use anyhow::Result;
use clap::Parser;
use portfolio_solver::{csv_parser, datastructures::*, portfolio_simulator};
use std::fs;

fn main() -> Result<()> {
    let args = ConfigArgs::parse();
    let config_path = args.config;
    let config_str = fs::read_to_string(config_path)?;
    let PortfolioExecutorConfig {
        files,
        portfolios,
        num_seeds,
        num_cores,
        out,
    } = serde_json::from_str(&config_str)?;

    DF_CONFIG.set(DataframeConfig::new()).ok();
    let df_config = DataframeConfig::global();
    let df = csv_parser::parse_hypergraph_dataframe(&files, None, num_cores)?
        .collect()?;
    let algorithms = csv_parser::extract_algorithm_columns(
        &df,
        &df_config.algorithm_fields,
    )?;
    let simulation = portfolio_simulator::simulation_df(
        &df,
        &algorithms,
        &portfolios,
        num_seeds,
        &["instance"],
        &df_config.algorithm_fields,
        num_cores,
    )?;
    csv_parser::df_to_csv_for_performance_profiles(
        simulation, df_config, out,
    )?;
    Ok(())
}
