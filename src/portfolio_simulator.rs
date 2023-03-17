use crate::datastructures::*;
use itertools::Itertools;
use polars::prelude::*;
use rand::prelude::*;

pub fn simulation_df(
    df: &DataFrame,
    algorithms: &ndarray::Array1<Algorithm>,
    portfolios: &[Portfolio],
    num_seeds: u32,
    instance_fields: &[&str],
    algorithm_fields: &[&str],
    num_cores: u32,
) -> LazyFrame {
    let portfolio_runs = portfolios
        .iter()
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
        .collect_vec();
    let algorithm_portfolios = simulate_algorithms_as_portfolio(
        df,
        algorithms,
        num_seeds,
        instance_fields,
        algorithm_fields,
        num_cores,
    );
    concat(
        &[portfolio_runs, vec![algorithm_portfolios]].concat(),
        false,
        false,
    )
    .expect("Failed to combine simulation dataframe")
}

fn simulate_portfolio_execution(
    df: &DataFrame,
    portfolio: &Portfolio,
    num_seeds: u32,
    instance_fields: &[&str],
    algorithm_fields: &[&str],
    num_cores: u32,
) -> LazyFrame {
    let runs = (0..num_seeds)
        .map(|seed| {
            let simulation_df = simulate(df, portfolio, seed as u64);
            portfolio_run_from_samples(
                simulation_df,
                instance_fields,
                algorithm_fields,
                num_cores,
                &portfolio.name,
            )
        })
        .collect_vec();
    concat(runs, false, false)
        .expect("Failed to combine portfolio simulations")
}

fn simulate_algorithms_as_portfolio(
    df: &DataFrame,
    algorithms: &ndarray::Array1<Algorithm>,
    num_seeds: u32,
    instance_fields: &[&str],
    algorithm_fields: &[&str],
    num_cores: u32,
) -> LazyFrame {
    let algorithm_portfolios = algorithms
        .iter()
        .map(|algo| Portfolio {
            name: algo.algorithm.clone(),
            resource_assignments: vec![(algo.clone(), num_cores as f64)],
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
        .collect_vec();
    concat(algorithm_portfolios, false, false)
        .expect("Failed to combine algorithm portfolio simulations")
}

fn simulate(df: &DataFrame, portfolio: &Portfolio, seed: u64) -> LazyFrame {
    let config = DataframeConfig::new();
    let explode_list = config
        .out_fields
        .iter()
        .filter(|s| !config.instance_fields.contains(s))
        .cloned()
        .collect_vec();
    let samples = &portfolio
        .resource_assignments
        .iter()
        .map(|(algo, cores)| {
            let num_samples = {
                let num_samples = cores / algo.num_threads as f64;
                if random::<f64>() >= num_samples - num_samples.floor() {
                    num_samples.floor()
                } else {
                    num_samples.ceil()
                }
            } as usize;
            df.clone()
                .lazy()
                .filter(col("algorithm").eq(lit(algo.algorithm.clone())))
                .filter(col("num_threads").eq(lit(algo.num_threads)))
                .groupby_stable(
                    config
                        .instance_fields
                        .iter()
                        .map(|f| col(f))
                        .collect_vec(),
                )
                .agg([col("*").sample_n(num_samples, true, true, Some(seed))])
                .explode(explode_list.clone())
                .with_column(lit(seed).alias("seed"))
        })
        .collect::<Vec<LazyFrame>>();
    concat(samples, false, false)
        .expect("Failed to build simulation dataframe")
}

fn portfolio_run_from_samples(
    df: LazyFrame,
    instance_fields: &[&str],
    algorithm_fields: &[&str],
    num_cores: u32,
    algorithm: &str,
) -> LazyFrame {
    df
        // .filter(col("feasibility_score").lt_eq(lit(0.03)))
        // .filter(col("failed").eq(lit("no")))
        // .filter(col("timeout").eq(lit("no")))
        .groupby(instance_fields)
        .agg([
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
mod tests {
    use polars::prelude::IntoLazy;

    use crate::{
        csv_parser::Data,
        datastructures::*,
        portfolio_simulator::{portfolio_run_from_samples, simulate},
        test_utils::*,
    };

    #[test]
    fn test_simple_model_simulation() {
        let config = Config {
            files: vec![
                "data/test/algo1.csv".to_string(),
                "data/test/algo2.csv".into(),
            ],
            ..default_config()
        };
        let data = Data::new(&config);
        let portfolio = Portfolio {
            name: "final_portfolio_opt".to_string(),
            resource_assignments: vec![
                (
                    Algorithm {
                        algorithm: "algo1".into(),
                        num_threads: 1,
                    },
                    0.0,
                ),
                (
                    Algorithm {
                        algorithm: "algo2".into(),
                        num_threads: 1,
                    },
                    2.0,
                ),
            ],
        };
        let simulation_df =
            simulate(&data.df, &portfolio, 42).collect().unwrap();
        assert_eq!(simulation_df.height(), 8);
        assert!(!simulation_df
            .column("algorithm")
            .unwrap()
            .utf8()
            .unwrap()
            .into_no_null_iter()
            .any(|s| s == "algo1"));
        let portfolio_df = portfolio_run_from_samples(
            simulation_df.lazy(),
            &["instance", "k", "feasibility_threshold"],
            &["algorithm", "num_threads"],
            2,
            "portfolio",
        )
        .collect()
        .unwrap();
        assert_eq!(portfolio_df.height(), 4);
        assert_eq!(
            portfolio_df
                .sort(["quality"], false)
                .unwrap()
                .column("quality")
                .unwrap()
                .f64()
                .unwrap()
                .to_ndarray()
                .unwrap(),
            ndarray::Array1::from_vec(vec![9.0, 11.0, 18.0, 24.0])
        );
    }
}
