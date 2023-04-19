use anyhow::Result;
use clap::Parser;
use log::info;
use std::fs;

use portfolio_solver::csv_parser;
use portfolio_solver::datastructures::*;
use portfolio_solver::solver;

mod mt_kahypar_parser;

fn main() -> Result<()> {
    let args = mt_kahypar_parser::Args::parse();
    env_logger::Builder::new()
        .filter_level(args.verbosity.log_level_filter())
        .init();
    let Ok(mt_kahypar_parser::Config {
        files,
        graphs,
        ks,
        feasibility_thresholds,
        num_cores,
        slowdown_ratio,
        num_seeds,
        out_dir,
        timeout,
    }) = mt_kahypar_parser::Config::from_cli(&args) else { std::process::exit(exitcode::CONFIG); };
    fs::create_dir(&out_dir).ok();
    let instance_filter = mt_kahypar_parser::InstanceFilter {
        instance_path: graphs,
        ks,
        feasibility_thresholds,
    };
    let df = mt_kahypar_parser::parse_hypergraph_dataframe(
        &files,
        Some(instance_filter),
        num_cores,
    )?;
    let data = csv_parser::Data::from_normalized_dataframe(
        df,
        num_cores,
        slowdown_ratio,
    )?;
    info!("{data}");
    let OptimizationResult {
        initial_portfolio,
        final_portfolio,
        gap: _,
    } = solver::solve(&data, num_cores as usize, timeout, None)?;
    info!("Final portfolio:\n{final_portfolio}");
    let random_portfolio = Portfolio::random(&data.algorithms, num_cores, 42);
    let portfolios = {
        let initial_portfolio_valid = match &initial_portfolio {
            Some(portfolio) => {
                portfolio.resource_assignments
                    != final_portfolio.resource_assignments
            }
            None => false,
        };
        let mut portfolios = vec![final_portfolio];
        if args.random_portfolio {
            portfolios.push(random_portfolio);
        }
        if args.initial_portfolio {
            if initial_portfolio_valid {
                portfolios.push(initial_portfolio.unwrap());
            } else {
                info!("The final portfolio is equal to the initial portfolio or no initial solution was provided. The initial portfolio will not be considered for portfolio execution.");
            }
        }
        portfolios
    };
    serde_json::to_writer_pretty(
        fs::File::create(out_dir.join("executor.json"))?,
        &mt_kahypar_parser::PortfolioExecutorConfig {
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
            fs::File::create(out_dir.join(portfolio_name + ".json"))?,
            &portfolio,
        )?;
    }
    Ok(())
}
