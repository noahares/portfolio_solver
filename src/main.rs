use std::env;
use anyhow::Result;

use portfolio_solver::csv_parser::Data;
use portfolio_solver::solver;
use portfolio_solver::portfolio_simulator;

fn main() -> Result<()> {
    let csv_path = env::args().nth(1).unwrap();
    let data = Data::new(&csv_path, 8)?;
    let result = solver::solve(&data, 8)?;
    Ok(())
}
