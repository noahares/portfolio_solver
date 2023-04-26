use anyhow::Result;
use clap::Parser;
use portfolio_solver::{csv_parser, portfolio_simulator};
use std::{fs, path::PathBuf};

#[path = "../mt_kahypar_parser.rs"]
mod mt_kahypar_parser;

#[derive(Parser)]
#[command(author, version, about)]
pub struct ConfigArgs {
    /// Path to the json config
    #[arg(short, long)]
    pub config: PathBuf,
}

fn main() -> Result<()> {
    let args = ConfigArgs::parse();
    let config_path = args.config;
    let config_str = fs::read_to_string(config_path)?;
    let mt_kahypar_parser::PortfolioExecutorConfig {
        files,
        portfolios,
        num_seeds,
        num_cores,
        out,
    } = serde_json::from_str(&config_str)?;

    let df =
        mt_kahypar_parser::parse_hypergraph_dataframe(&files, None, num_cores)
            .or_else(|_| {
                csv_parser::parse_normalized_csvs(&files, None, num_cores)
            })?
            .collect()?;
    let algorithms = csv_parser::extract_algorithm_columns(&df)?;
    let simulation = portfolio_simulator::simulation_df(
        &df,
        &algorithms,
        &portfolios,
        num_seeds,
        &["instance"],
        &["algorithm", "num_threads"],
        num_cores,
    )?;
    csv_parser::df_to_normalized_csv(simulation, out)?;
    Ok(())
}
