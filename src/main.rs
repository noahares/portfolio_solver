use anyhow::Result;
use clap::Parser;
use std::fs;

use portfolio_solver::csv_parser;
use portfolio_solver::datastructures::*;
use portfolio_solver::portfolio_simulator;
use portfolio_solver::solver;

fn main() -> Result<()> {
    let args = Args::parse();
    let config_path = args.config;
    let config_str = fs::read_to_string(config_path)
        .expect("Provided config file does not exist");
    let config = {
        let mut config: Config = serde_json::from_str(&config_str)
            .expect("Error while reading config file");
        if let Some(slowdown_ratio) = args.slowdown_ratio {
            config.slowdown_ratio = slowdown_ratio;
        }
        if config.slowdown_ratio == 0.0 {
            config.slowdown_ratio = std::u32::MAX as f64;
        }
        if let Some(out_dir) = args.out_dir.as_deref() {
            config.out_dir = out_dir
                .to_str()
                .expect("output directory not found")
                .to_string();
        }
        if let Some(timeout) = args.timeout {
            config.timeout = timeout;
        }
        config
    };
    let k = config.num_cores;
    let files = config.files.clone();
    let num_seeds = config.num_seeds;
    let out_dir = config.out_dir.trim_end_matches('/').to_owned();
    let num_cores = config.num_cores;
    let timeout = config.timeout.clone();
    fs::create_dir(&out_dir).ok();
    let data = csv_parser::Data::new(&config);
    let df_config = DataframeConfig::new();
    println!("{data}");
    let portfolio = solver::solve(&data, k as usize, timeout);
    println!("{portfolio}");
    let portfolio_simulation = portfolio_simulator::simulation_df(
        &data.df,
        &data.algorithms,
        &portfolio,
        num_seeds,
        &df_config.instance_fields,
        &df_config.algorithm_fields,
        num_cores,
    );
    csv_parser::df_to_csv_for_performance_profiles(
        portfolio_simulation,
        &df_config,
        &(out_dir.to_owned() + "/simulation.csv"),
    );
    serde_json::to_writer_pretty(
        fs::File::create(out_dir.to_owned() + "/executor.json")?,
        &PortfolioExecutorConfig {
            files,
            portfolio: portfolio.clone(),
            num_seeds,
            num_cores,
            out: out_dir.to_owned() + "/execution.csv",
        },
    )?;
    serde_json::to_writer_pretty(
        fs::File::create(out_dir + "/portfolio.json")?,
        &portfolio,
    )?;
    Ok(())
}
