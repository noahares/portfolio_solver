use anyhow::Result;
use clap::Parser;
use itertools::Itertools;
use polars::prelude::*;
use portfolio_solver::csv_parser;
use portfolio_solver::datastructures::*;
use std::fs;

fn main() -> Result<()> {
    let args = ConfigArgs::parse();
    let config_path = args.config;
    let config_str = fs::read_to_string(config_path)?;
    let config: QualityLowerBoundConfig = serde_json::from_str(&config_str)?;
    let paths = config.files;

    DF_CONFIG.set(DataframeConfig::new()).ok();
    let df_config = DataframeConfig::global();
    let df = csv_parser::preprocess_df(&paths, df_config)?.collect()?;
    let mut best_per_instance = csv_parser::best_per_instance(
        df.lazy(),
        &df_config.instance_fields,
        "quality",
    )
    .select(
        [
            df_config
                .instance_fields
                .iter()
                .map(|f| col(f))
                .collect_vec(),
            vec![col("best_quality").alias("quality_lb")],
        ]
        .concat(),
    )
    .collect()?;
    let mut file = fs::File::create(config.out)?;
    CsvWriter::new(&mut file).finish(&mut best_per_instance)?;
    Ok(())
}

// fn get_quality_lb(config: &csv_parser::DataframeConfig) -> Result<LazyFrame> {
// }
//
// #[cfg(test)]
// mod tests {
//     #[test]
//     fn test_quality_lb() {
//     }
// }
