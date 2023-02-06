use crate::{csv_parser::Data, datastructures::*};
use itertools::Itertools;
use polars::prelude::*;

pub fn simulate(data: &Data, portfolio: SolverResult, seed: u64) -> LazyFrame {
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
        })
        .collect::<Vec<LazyFrame>>();
    concat(samples, false, false)
        .expect("Failed to build simulation dataframe")
}

#[cfg(test)]
mod tests {
    use crate::{
        csv_parser::Data, datastructures::*, portfolio_simulator::simulate,
    };

    #[test]
    fn test_simple_model_simulation() {
        let config = Config {
            files: vec![
                "data/test/algo1.csv".to_string(),
                "data/test/algo2.csv".into(),
            ],
            quality_lb: "data/test/quality_lb.csv".to_string(),
            num_cores: 2,
            slowdown_ratio: std::f64::MAX,
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
        let simulation_df = simulate(&data, portfolio, 42).collect().unwrap();
        assert_eq!(simulation_df.height(), 8);
        assert!(!simulation_df
            .column("algorithm")
            .unwrap()
            .utf8()
            .unwrap()
            .into_no_null_iter()
            .any(|s| s == "algo1"));
    }
}
