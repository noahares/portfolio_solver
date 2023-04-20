#![warn(missing_docs)]
//! Optimize a parallel algorithm portfolio for a set of instances.
//!
//! Provides a solver for the algorithm portfolio optimization problem and helper modules for csv
//! parsing and portfolio simulation.
//!
//! The problem definition and corresponding linear program is found here: TODO:
//!
//! Requirements: Gurobi installation (9.0 or higher) and
//! [license](http://www.gurobi.com/downloads/licenses/license-center).
//! Don't forget to set the environment variable `GUROBI_HOME` to the installation path of Gurobi.
//!
//! This project also contains 2 executables that use the library to optimize and simulate a portfolio for
//! hypergraph partitioning. Their usage is documented
//! [here](https://github.com/noahares/portfolio_solver).
//!
//! The code of these executables will be used to show how the library can
//! be applied to solve such an optimization problem and is intended as a starting point for
//! using it with other input data. Most importantly, parsing of normalized data frames is
//! supported, so all you need to do is convert your data into this (very simple) format.
//!
//! Example
//! ```rust
//! use portfolio_solver::csv_parser;
//! use portfolio_solver::datastructures;
//! use portfolio_solver::solver;
//! # use std::path::PathBuf;
//! # use anyhow::Result;
//!
//! fn example() -> Result<()> {
//!     let paths = [PathBuf::from("input1.csv"), "input2.csv".into()];
//!     let num_cores: u32 = 8; // number of cores available to the portfolio
//!     let slowdown_ratio = 0.5; // find a portfolio that is approximatly 2x faster than the best
//!     let timeout = datastructures::Timeout::default(); // defaults to 900 seconds
//!
//!     // normalized csvs have the following header (types in parenthesis):
//!     // algorithm(str),num_threads(int),instance(str),quality(float),time(float),valid(bool)
//!     let df = csv_parser::parse_normalized_csvs(
//!         &paths,
//!         None, // optionally provide the path to a csv file with instance names to filter for
//!         num_cores,
//!         )?;
//!
//!     let data = csv_parser::Data::from_normalized_dataframe(
//!         df,
//!         num_cores,
//!         slowdown_ratio,
//!        )?;
//!
//!     let datastructures::OptimizationResult {
//!         initial_portfolio: _,
//!         final_portfolio,
//!         gap: _,
//!         } = solver::solve(
//!                 &data,
//!                 num_cores as usize,
//!                 timeout,
//!                 None, // optionally provide a initial solutions, fallback to a heuristic
//!                 )?;
//!
//!     // datastructures::Portfolio implements serde::{Serialize, Deserialize}
//!     // Especially useful for reading portfolios back from json for simulation
//!     let output = serde_json::to_string(&final_portfolio)?;
//!     println!("{}", output);
//!     Ok(())
//! }
//!
//! ```

/// Various helpers for csv parsing of normalized dataframes and creating the input for the
/// solver.
pub mod csv_parser;

/// Data structures for easier usage of the solver.
pub mod datastructures;

/// Helper functions to simulate a portfolio execution from csv data.
pub mod portfolio_simulator;

/// A solver based on Gurobi for the algorithm portfolio optimization problem.
pub mod solver;
