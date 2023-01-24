use serde::Deserialize;

pub type Instance = String;
pub type Algorithm = String;

#[derive(Debug, Default, Deserialize, Clone)]
pub struct Datapoint {
    pub algorithm: Algorithm,
    #[serde(rename = "graph")]
    pub instance: Instance,
    #[serde(rename = "imbalance")]
    pub feasibility_score: f64,
    #[serde(rename = "km1")]
    pub quality: f64,
    #[serde(rename = "partitionTime")]
    pub time: f64,
}

pub struct Dataframe {
    pub datapoints: Vec<Datapoint>,
    pub algorithms: Vec<Algorithm>,
    pub instances: Vec<Instance>,
}

pub struct Statistics {
    pub mean: f64,
    pub stddev: f64,
}

// fn preprocess_dataframe(dataframe: &Dataframe) -> Result<HashMap<String, f64>> {
//     let grouped_by_instance = dataframe
//         .datapoints
//         .iter()
//         .group_by(|datapoint| &datapoint.instance);
//
//     let best_quality_per_instance: HashMap<Instance, f64> = grouped_by_instance
//         .into_iter()
//         .map(|(instance, group)| (instance.clone(), group.min_by(|&a, &b| a.quality.partial_cmp(&b.quality).unwrap()).unwrap().quality))
//         .collect();
//
//     let statistics: HashMap<(Instance, Algorithm), Statistics> = HashMap::new();
//     for (instance, group) in &grouped_by_instance {
//         for (algorithm, runs) in &group.into_iter().group_by(|group| &group.algorithm) {
//             statistics.insert((instance.clone(), algorithm.clone()), Statistics { mean: mean_quality(runs), stddev: 0.0 });
//         }
//     }
//
//
//     Ok(best_quality_per_instance)
// }
//
// fn mean_quality(datapoints: &[Datapoint]) -> f64 {
//     datapoints.iter().fold(0.0, |acc, datapoint| acc + datapoint.quality) / datapoints.len() as f64
// }
