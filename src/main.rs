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
    let config = {
        let Ok(config) = Config::from_cli(&args) else { std::process::exit(exitcode::CONFIG); };
        CONFIG.set(config).ok();
        Config::global()
    };
    let k = config.num_cores;
    let files = config.files.clone();
    let num_seeds = config.num_seeds;
    let out_dir = &config.out_dir;
    let num_cores = config.num_cores;
    let timeout = config.timeout.clone();
    fs::create_dir(out_dir).ok();
    let data = csv_parser::Data::new()?;
    DF_CONFIG.set(DataframeConfig::new()).ok();
    let df_config = DataframeConfig::global();
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
        df_config,
        out_dir.join("simulation.csv"),
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
        fs::File::create(out_dir.join("executor.json"))?,
        &PortfolioExecutorConfig {
            files,
            portfolios: portfolios.clone(),
            num_seeds,
            num_cores,
            out: out_dir.join("execution.csv"),
        },
    )?;
    for portfolio in portfolios {
        let portfolio_name = portfolio.name.replace("_opt", "");
        serde_json::to_writer_pretty(
            fs::File::create(
                out_dir
                    .join("/".to_owned() + portfolio_name.as_str() + ".json"),
            )?,
            &portfolio,
        )?;
    }
    Ok(())
}
