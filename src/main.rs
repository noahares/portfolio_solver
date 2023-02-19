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
    let files = config.files.clone();
    let num_seeds = config.num_seeds;
    let out_dir = config.out_dir.trim_end_matches('/').to_owned();
    let num_cores = config.num_cores;
    fs::create_dir(&out_dir).ok();
    let data = csv_parser::Data::new(&config);
    let df_config = DataframeConfig::new();
    println!("{data}");
    let portfolio = solver::solve(&data, k as usize);
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
