use crate::{csv_parser::Data, datastructures::*};
use itertools::Itertools;
use polars::prelude::*;

pub fn simulate_portfolio_execution(
    data: &Data,
    portfolio: SolverResult,
    num_seeds: u32,
    instance_fields: &[&str],
) -> DataFrame {
    let num_cores = portfolio
        .resource_assignments
        .iter()
        .fold(0, |acc, a| acc + (a.1) as u32 * a.0.num_threads);
    let runs = (0..num_seeds)
        .map(|seed| {
            let simulation_df = simulate(data, &portfolio, seed as u64);
            portfolio_run_from_samples(
                simulation_df,
                instance_fields,
                num_cores,
            )
        })
        .collect_vec();
    concat(runs, false, false)
        .expect("Failed to combine portfolio simulations")
        .collect()
        .unwrap()
}

fn simulate(data: &Data, portfolio: &SolverResult, seed: u64) -> LazyFrame {
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
            data.df
                .clone()
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
                .agg([col("*").sample_n(
                    cores.floor() as usize,
                    false,
                    true,
                    Some(seed),
                )])
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
    num_cores: u32,
) -> LazyFrame {
    df.filter(col("feasibility_score").lt_eq(lit(0.03)))
        .filter(col("failed").eq(lit("no")))
        .filter(col("timeout").eq(lit("no")))
        .groupby(instance_fields)
        .agg([
            lit("portfolio-solver").alias("algorithm"),
            lit(num_cores).alias("num_threads"),
            lit(0.03).alias("epsilon"),
            col("*")
                .exclude(
                    [
                        instance_fields,
                        &["algorithm", "quality", "time", "num_threads"],
                    ]
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
    };

    fn default_config() -> Config {
        Config {
            files: vec![],
            quality_lb: "data/test/quality_lb.csv".to_string(),
            num_cores: 2,
            slowdown_ratio: std::f64::MAX,
            num_seeds: 1,
            out_file: "".to_string(),
        }
    }

    #[test]
    fn test_simple_model_simulation() {
        let config = Config {
            files: vec![
                "data/test/algo1.csv".to_string(),
                "data/test/algo2.csv".into(),
            ],
            ..default_config()
        };
        let data = Data::new(config);
        let portfolio = SolverResult {
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
        let simulation_df = simulate(&data, &portfolio, 42).collect().unwrap();
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
            &["instance", "k"],
            2,
        );
        // println!("{portfolio_df}");
        // println!("{:?}", portfolio_df.get_column_names());
        // assert_eq!(0, 1);
        // TODO:  better assertions and new test algorithm
    }
}
