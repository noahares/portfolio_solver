use std::env;
use anyhow::Result;

use portfolio_solver::csv_parser;
use portfolio_solver::solver;
use portfolio_solver::portfolio_simulator;

fn main() -> Result<()> {
    let csv_path = env::args().nth(1).unwrap();
    let dataframe = csv_parser::read_kahypar_csv(&csv_path, 8)?;
    // let result = solver::solve(&dataframe, 8)?;
    Ok(())
}
