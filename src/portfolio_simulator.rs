use crate::datastructures::*;
use anyhow::Result;
use itertools::Itertools;
use polars::prelude::*;
use rand::prelude::*;

/// Simulate execution of a portfolio
///
/// For each algorithm `num_seeds` runs will be sampled from the data frame for each instance
pub fn simulation_df(
    df: &DataFrame,
    algorithms: &ndarray::Array1<Algorithm>,
    portfolios: &[Portfolio],
    num_seeds: u32,
    instance_fields: &[&str],
    algorithm_fields: &[&str],
    num_cores: u32,
) -> Result<LazyFrame> {
    let portfolio_runs = portfolios
        .iter()
        .filter(|p| !p.resource_assignments.is_empty())
        .map(|p| {
            simulate_portfolio_execution(
                df,
                p,
                num_seeds,
                instance_fields,
                algorithm_fields,
                num_cores,
            )
        })
        .filter_map(Result::ok)
        .collect_vec();
    let algorithm_portfolios = simulate_algorithms_as_portfolio(
        df,
        algorithms,
        num_seeds,
        instance_fields,
        algorithm_fields,
        num_cores,
    )?;
    Ok(concat(
        &[portfolio_runs, vec![algorithm_portfolios]].concat(),
        false,
        false,
    )?)
}

fn simulate_portfolio_execution(
    df: &DataFrame,
    portfolio: &Portfolio,
    num_seeds: u32,
    instance_fields: &[&str],
    algorithm_fields: &[&str],
    num_cores: u32,
) -> Result<LazyFrame> {
    let runs = (0..num_seeds)
        .map(|seed| -> Result<LazyFrame> {
            let simulation_df = simulate(df, portfolio, seed as u64)?;
            Ok(portfolio_run_from_samples(
                simulation_df,
                instance_fields,
                algorithm_fields,
                num_cores,
                &portfolio.name,
            ))
        })
        .filter_map(Result::ok)
        .collect_vec();
    Ok(concat(runs, false, false)?)
}

fn simulate_algorithms_as_portfolio(
    df: &DataFrame,
    algorithms: &ndarray::Array1<Algorithm>,
    num_seeds: u32,
    instance_fields: &[&str],
    algorithm_fields: &[&str],
    num_cores: u32,
) -> Result<LazyFrame> {
    let algorithm_portfolios = algorithms
        .iter()
        .filter(|a| a.num_threads <= num_cores)
        .map(|algo| {
            let num_samples = {
                let num_samples = num_cores as f64 / algo.num_threads as f64;
                if random::<f64>() >= num_samples - num_samples.floor() {
                    num_samples.floor()
                } else {
                    num_samples.ceil()
                }
            };
            Portfolio {
                name: algo.algorithm.clone()
                    + " "
                    + algo.num_threads.to_string().as_str(),
                resource_assignments: vec![(algo.clone(), num_samples)],
            }
        })
        .map(|portfolio| {
            simulate_portfolio_execution(
                df,
                &portfolio,
                num_seeds,
                instance_fields,
                algorithm_fields,
                num_cores,
            )
        })
        .filter_map(Result::ok)
        .collect_vec();
    Ok(concat(algorithm_portfolios, false, false)?)
}

fn simulate(
    df: &DataFrame,
    portfolio: &Portfolio,
    seed: u64,
) -> Result<LazyFrame> {
    let explode_list =
        vec!["algorithm", "num_threads", "quality", "time", "valid"];
    let samples = &portfolio
        .resource_assignments
        .iter()
        .map(|(algo, cores)| {
            df.clone()
                .lazy()
                .filter(col("algorithm").eq(lit(algo.algorithm.clone())))
                .filter(col("num_threads").eq(lit(algo.num_threads)))
                .groupby_stable([col("instance")])
                .agg([col("*").sample_n(
                    *cores as usize,
                    true,
                    true,
                    Some(seed),
                )])
                .explode(explode_list.clone())
                .with_column(lit(seed).alias("seed"))
        })
        .collect::<Vec<LazyFrame>>();
    Ok(concat(samples, false, false)?)
}

fn portfolio_run_from_samples(
    df: LazyFrame,
    instance_fields: &[&str],
    algorithm_fields: &[&str],
    num_cores: u32,
    algorithm: &str,
) -> LazyFrame {
    df.groupby(instance_fields).agg([
        lit(algorithm).alias("algorithm"),
        lit(num_cores).alias("num_threads"),
        col("*")
            .exclude(
                [instance_fields, algorithm_fields, &["quality", "time"]]
                    .concat(),
            )
            .sort_by(vec![col("quality")], vec![false])
            .first(),
        min("quality"),
        max("time"),
    ])
}

#[cfg(test)]
mod tests;
