use anyhow::Result;
use clap::Parser;
use clap_verbosity_flag::Verbosity;
use itertools::Itertools;
use log::warn;
use polars::{lazy::dsl::GetOutput, prelude::*};
use portfolio_solver::datastructures::{Portfolio, Timeout};
use serde::{Deserialize, Serialize};
use std::{f64::EPSILON, fs, path::PathBuf};

#[derive(Serialize, Deserialize, Clone)]
pub struct Config {
    pub files: Vec<PathBuf>,
    #[serde(default)]
    pub graphs: PathBuf,
    #[serde(default = "default_ks")]
    pub ks: Vec<i64>,
    #[serde(default = "default_feasibility_thresholds")]
    pub feasibility_thresholds: Vec<f64>,
    pub num_cores: u32,
    pub slowdown_ratio: f64,
    pub num_seeds: u32,
    pub out_dir: PathBuf,
    #[serde(default)]
    pub timeout: Timeout,
}

#[derive(Serialize, Deserialize)]
pub struct PortfolioExecutorConfig {
    pub files: Vec<PathBuf>,
    pub portfolios: Vec<Portfolio>,
    pub num_seeds: u32,
    pub num_cores: u32,
    pub out: PathBuf,
}

impl Config {
    pub fn from_cli(args: &Args) -> Result<Config> {
        let config_path = &args.config;
        let config_str = fs::read_to_string(config_path)?;
        let mut config: Config = serde_json::from_str(&config_str)?;
        if let Some(slowdown_ratio) = args.slowdown_ratio {
            config.slowdown_ratio = slowdown_ratio;
        }
        if config.slowdown_ratio == 0.0 {
            config.slowdown_ratio = std::u32::MAX as f64;
        }
        if let Some(out_dir) = &args.out_dir {
            config.out_dir = out_dir.to_path_buf();
        }
        if let Some(timeout) = &args.timeout {
            config.timeout = timeout.clone();
        }
        if let Some(num_cores) = args.num_cores {
            config.num_cores = num_cores;
        }
        if let Some(num_seeds) = args.num_seeds {
            config.num_seeds = num_seeds;
        }
        if let Some(graphs) = &args.graphs {
            config.graphs = graphs.to_path_buf();
        }
        if let Some(files) = &args.files {
            config.files = files.to_vec();
        }
        if let Some(ks) = &args.ks {
            config.ks = ks.to_vec();
        }
        if let Some(feasibility_thresholds) = &args.feasibility_thresholds {
            config.feasibility_thresholds = feasibility_thresholds.to_vec();
        }
        Ok(config)
    }
}

fn default_ks() -> Vec<i64> {
    vec![2, 4, 8, 16, 32, 64, 128]
}

fn default_feasibility_thresholds() -> Vec<f64> {
    vec![0.03]
}

pub struct InstanceFilter {
    pub instance_path: PathBuf,
    pub ks: Vec<i64>,
    pub feasibility_thresholds: Vec<f64>,
}

pub fn parse_hypergraph_dataframe(
    paths: &[PathBuf],
    desired_instances: Option<InstanceFilter>,
    num_cores: u32,
) -> Result<LazyFrame> {
    let instance_fields: [Expr; 3] = [col("graph"), col("k"), col("epsilon")];
    let read_df = |path: &PathBuf,
                   in_fields: &[&'static str]|
     -> Result<LazyFrame> {
        let mut dataframe = CsvReader::from_path(path)?
            .with_comment_char(Some(b'#'))
            .has_header(true)
            .with_columns(Some(
                in_fields.iter().map(|s| s.to_string()).collect_vec(),
            ))
            .with_dtypes(Some(&Schema::from(
                [Field::new("km1", DataType::Float64)].into_iter(),
            )))
            .finish()?
            .lazy()
            .filter(col("num_threads").lt_eq(lit(num_cores)))
            .with_columns([
                col("graph").apply(
                    |s: Series| {
                        Ok(s.utf8()?
                            .into_no_null_iter()
                            .map(fix_instance_names)
                            .collect())
                    },
                    GetOutput::from_type(DataType::Utf8),
                ),
                col("km1").apply(
                    |s: Series| {
                        Ok(s.f64()?
                            .into_no_null_iter()
                            .map(|i| if i.abs() <= EPSILON { 1.0 } else { i })
                            .collect())
                    },
                    GetOutput::from_type(DataType::Float64),
                ),
            ]);
        match &desired_instances {
            Some(filter) => {
                if let Ok(instance_filter) = get_desired_instances(
                    &filter.instance_path,
                    &filter.ks,
                    &filter.feasibility_thresholds,
                ) {
                    dataframe = dataframe.join(
                        instance_filter,
                        &instance_fields,
                        &instance_fields,
                        JoinType::Inner,
                    );
                }
            }
            None => (),
        };
        Ok(dataframe.select([
            concat_str(&instance_fields, "").alias("instance"),
            col("algorithm"),
            col("num_threads"),
            col("km1").alias("quality"),
            col("totalPartitionTime").alias("time"),
            col("imbalance")
                .lt_eq(col("epsilon"))
                .and(col("failed").eq(lit("no")))
                .and(col("timeout").eq(lit("no")))
                .alias("valid"),
        ]))
    };

    let columns: [&str; 10] = [
        "algorithm",
        "num_threads",
        "graph",
        "k",
        "epsilon",
        "imbalance",
        "km1",
        "totalPartitionTime",
        "failed",
        "timeout",
    ];
    let mut fixed_in_fields = columns.to_vec();
    fixed_in_fields.retain(|s| *s != "num_threads");
    let dataframes: Vec<LazyFrame> = paths
        .iter()
        .map(|path| match read_df(path, &columns) {
            Ok(result) => Ok(result),
            Err(_) => match read_df(path, &fixed_in_fields) {
                Ok(result) => {
                    Ok(result.with_column(lit(1_i64).alias("num_threads")))
                }
                Err(err) => anyhow::bail!(err),
            },
        })
        .filter_map(Result::ok)
        .collect();
    match dataframes.is_empty() {
        true => anyhow::bail!("Failed to parse data frames"),
        false => concat(dataframes, true, true).map_err(anyhow::Error::from),
    }
}

fn get_desired_instances(
    graphs_path: &PathBuf,
    num_parts: &Vec<i64>,
    feasibility_thresholds: &Vec<f64>,
) -> Result<LazyFrame> {
    if let Ok(reader) = CsvReader::from_path(graphs_path) {
        let graph_df = reader.has_header(true).finish()?.lazy();
        let k_df = df! {
            "k" => num_parts
        }?;
        let eps_df = df! {
            "epsilon" => feasibility_thresholds
        }?;
        Ok(graph_df.cross_join(k_df.lazy()).cross_join(eps_df.lazy()))
    } else {
        warn!(
            "Provided graph file: {:?} not found, using all graphs",
            graphs_path
        );
        Err(anyhow::Error::msg("No graph file"))
    }
}

fn fix_instance_names(instance: &str) -> String {
    if instance.ends_with("scotch") {
        instance.replace("scotch", "graph")
    } else {
        instance.to_string()
    }
}

#[derive(Parser)]
#[command(author, version, about)]
pub struct Args {
    /// Path to the json config
    #[arg(short, long)]
    pub config: PathBuf,
    /// List of CSV files containing the input data
    #[arg(short, long, value_delimiter = ' ', num_args = 0..)]
    pub files: Option<Vec<PathBuf>>,
    /// Filter instances by number of blocks (k)
    #[arg(long, value_name = "k", value_delimiter = ' ', num_args = 0..)]
    pub ks: Option<Vec<i64>>,
    /// Filter instances by feasibility threshold (epsilon)
    #[arg(long, value_name = "e", value_delimiter = ' ', num_args = 0..)]
    pub feasibility_thresholds: Option<Vec<f64>>,
    /// Path to a CSV file containing a list of graphs
    #[arg(short, long, value_name = "FILE")]
    pub graphs: Option<PathBuf>,
    /// Filter algorithms to get a portfolio with gmean-expected slowdown
    /// (Values < 1.0 mean speedup)
    #[arg(short, long)]
    pub slowdown_ratio: Option<f64>,
    /// How often a portfolio run is sampled for each instance
    #[arg(short, long)]
    pub num_seeds: Option<u32>,
    /// Path to the output directory
    #[arg(short, long, value_name = "DIR")]
    pub out_dir: Option<PathBuf>,
    /// Timeout for the LP solver in seconds
    #[arg(short, long, value_parser)]
    pub timeout: Option<Timeout>,
    /// Number of cores available to the portfolio
    #[arg(short = 'k', long)]
    pub num_cores: Option<u32>,
    /// Write initial portfolio to output
    /// (Only if different from final portfolio)
    #[arg(short, long)]
    pub initial_portfolio: bool,
    /// Write random portfolio to output
    /// (Only if at least 1 sequential algorithm remains after slowdown filtering)
    #[arg(short, long)]
    pub random_portfolio: bool,
    #[command(flatten)]
    pub verbosity: Verbosity,
}

#[cfg(test)]
mod tests {
    use super::parse_hypergraph_dataframe;
    use polars::prelude::*;
    use std::path::PathBuf;

    #[test]
    fn test_hypergraph_parser() {
        let k = 4;
        let path = PathBuf::from("data/test/algo4.csv");
        let df = parse_hypergraph_dataframe(&[path], None, k)
            .unwrap()
            .collect()
            .unwrap();
        assert_eq!(df.height(), 12);
        assert_eq!(
            df["valid"],
            Series::new(
                "valid",
                &[
                    true, false, true, true, false, true, true, false, true,
                    true, false, true
                ]
            )
        );
    }
}
