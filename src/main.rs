use anyhow::Result;
use std::env;

use portfolio_solver::csv_parser::Data;
use portfolio_solver::portfolio_simulator;
use portfolio_solver::solver;

fn main() -> Result<()> {
    let csv_path = env::args().nth(1).unwrap();
    let num_cores = env::args().nth(2).unwrap().parse().unwrap();
    let data = Data::new(&csv_path, num_cores)?;
    let result = solver::solve(&data, num_cores as usize)?;
    println!("{result:?}");
    println!("{:?}", data.algorithms);
    Ok(())
}
