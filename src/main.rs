use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use log::info;
use std::fs;

use portfolio_solver::csv_parser;
use portfolio_solver::datastructures::*;
use portfolio_solver::portfolio_simulator;
use portfolio_solver::solver;

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    let Ok(config) = parse_config(&args) else { std::process::exit(exitcode::CONFIG); };
    let k = config.num_cores;
    let files = config.files.clone();
    let num_seeds = config.num_seeds;
    let out_dir = config.out_dir.trim_end_matches('/').to_owned();
    let num_cores = config.num_cores;
    let timeout = config.timeout.clone();
    fs::create_dir(&out_dir).ok();
    let data = csv_parser::Data::new(&config)?;
    let df_config = DataframeConfig::new();
    info!("{data}");
    let OptimizationResult {
        initial_portfolio,
        final_portfolio,
        gap: _,
    } = solver::solve(&data, k as usize, timeout)?;
    info!("Final portfolio:\n{final_portfolio}");
    let random_portfolio = Portfolio::random(&data.algorithms, num_cores, 42);
    let portfolio_simulation = portfolio_simulator::simulation_df(
        &data.df,
        &data.algorithms,
        &[final_portfolio.clone()],
        num_seeds,
        &df_config.instance_fields,
        &df_config.algorithm_fields,
        num_cores,
    )?;
    csv_parser::df_to_csv_for_performance_profiles(
        portfolio_simulation,
        &df_config,
        &(out_dir.to_owned() + "/simulation.csv"),
    )?;
    let portfolios = {
        let mut portfolios = vec![final_portfolio];
        if args.random_portfolio {
            portfolios.push(random_portfolio);
        }
        if args.initial_portfolio {
            portfolios.push(initial_portfolio);
        }
        portfolios
    };
    serde_json::to_writer_pretty(
        fs::File::create(out_dir.to_owned() + "/executor.json")?,
        &PortfolioExecutorConfig {
            files,
            portfolios: portfolios.clone(),
            num_seeds,
            num_cores,
            out: out_dir.to_owned() + "/execution.csv",
        },
    )?;
    for portfolio in portfolios {
        let portfolio_name = portfolio.name.replace("_opt", "");
        serde_json::to_writer_pretty(
            fs::File::create(
                out_dir.clone() + "/" + portfolio_name.as_str() + ".json",
            )?,
            &portfolio,
        )?;
    }
    Ok(())
}

fn parse_config(args: &Args) -> Result<Config> {
    let config_path = &args.config;
    let config_str = fs::read_to_string(config_path)?;
    let mut config: Config = serde_json::from_str(&config_str)?;
    if let Some(slowdown_ratio) = args.slowdown_ratio {
        config.slowdown_ratio = slowdown_ratio;
    }
    if config.slowdown_ratio == 0.0 {
        config.slowdown_ratio = std::u32::MAX as f64;
    }
    if let Some(out_dir) = args.out_dir.as_deref() {
        config.out_dir = out_dir
            .to_str()
            .context(format!("{:?} is not a valid directory", out_dir))?
            .to_string();
    }
    if let Some(timeout) = &args.timeout {
        config.timeout = timeout.clone();
    }
    if let Some(num_cores) = args.num_cores {
        config.num_cores = num_cores;
    }
    Ok(config)
}
