use std::collections::HashSet;

use anyhow::Result;
use crate::datastructures::*;

pub fn read_csv(path: &str) -> Result<Dataframe> {
    let mut datapoints: Vec<Datapoint> = Vec::new();
    let mut algorithms = HashSet::new();
    let mut instances = HashSet::new();
    let mut rdr = csv::ReaderBuilder::new()
        .comment(Some(b'#'))
        .from_path(path)?;
    for record in rdr.deserialize() {
        let datapoint: Datapoint = record?;
        algorithms.insert(datapoint.algorithm.clone());
        instances.insert(datapoint.instance.clone());
        datapoints.push(datapoint);
    }
    Ok(Dataframe { datapoints, algorithms: Vec::from_iter(algorithms), instances: Vec::from_iter(instances) })
}
