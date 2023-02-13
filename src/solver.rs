use crate::datastructures::*;
use itertools::Itertools;

use crate::csv_parser::Data;
use grb::prelude::*;
use ndarray::{Array1, Array2, Array3};

pub fn solve(data: &Data, num_cores: usize) -> SolverResult {
    let mut model =
        Model::new("portfolio_model").expect("Failed to create Gurobi Model");
    model.set_param(param::NumericFocus, 1).unwrap();
    // model.set_param(param::SolFiles, "portfolio_model".to_string())?;
    model.set_param(param::TimeLimit, 900.0).unwrap();
    let (n, m) = (data.num_algorithms, data.num_instances);

    let a =
        Array3::<grb::Var>::from_shape_fn((m, n, num_cores), |(i, j, k)| {
            add_binvar!(model, name: format!("a_{i}_{j}_{k}").as_str())
                .unwrap()
        });
    let b = Array2::<grb::Var>::from_shape_fn((n, num_cores), |(j, k)| {
        add_binvar!(model, name: format!("b_{j}_{k}").as_str()).unwrap()
    });
    let q = Array1::<grb::Var>::from_shape_fn(m, |i| {
        add_ctsvar!(model, name: format!("q_{i}").as_str(), bounds: 0..)
            .unwrap()
    });
    let best_per_instance = &data.best_per_instance;

    let e_min = &data.stats;

    // constraint 1
    let _c_1 = a
        .indexed_iter()
        .map(|((i, j, k), &val_a)| {
            model.add_constr(
                format!("c1_{i}_{j}_{k}").as_str(),
                c!(val_a * e_min[(i, j, k)] <= q[i]),
            )
        })
        .collect_vec();

    // constraint 2
    let _c_2 = b
        .rows()
        .into_iter()
        .enumerate()
        .map(|(i, row)| {
            model.add_constr(
                format!("c2_{i}").as_str(),
                c!(row.into_iter().grb_sum() <= 1),
            )
        })
        .collect_vec();

    // constraint 3
    let sums = b
        .rows()
        .into_iter()
        .zip(&data.algorithms)
        .map(|(row, algo)| {
            row.into_iter()
                .zip(1..=num_cores)
                .map(|(var, k)| *var * k * algo.num_threads)
                .grb_sum()
        })
        .grb_sum();
    let _c_3 = model.add_constr("c3", c!(sums == num_cores));

    // constraint 4
    let _c_4 = a
        .outer_iter()
        .into_iter()
        .enumerate()
        .map(|(i, row)| {
            model.add_constr(
                format!("c4_{i}").as_str(),
                c!(row.iter().grb_sum() == 1),
            )
        })
        .collect_vec();

    // constraint 5
    let _c_5 = a
        .indexed_iter()
        .map(|((i, j, k), &val_a)| {
            model.add_constr(
                format!("c5_{i}_{j}_{k}").as_str(),
                c!(val_a <= b[(j, k)]),
            )
        })
        .collect_vec();

    let objective_function = q
        .iter()
        .zip(best_per_instance.iter())
        .map(|(&var, &best)| var * (1.0 / best))
        .grb_sum();

    let mut callback = |w: Where| {
        if let Where::MIPSol(ctx) = w {
            let sol = ctx.get_solution(b.iter())?;
            let obj = ctx.obj()?;
            let obj_bnd = ctx.obj_bnd()?;
            let res =
                postprocess_solution(sol, n, num_cores, &data.algorithms);
            println!("{res}{}", 1.0 - (obj_bnd / obj));
        }
        Ok(())
    };

    let start_vals = data
        .best_per_instance_count
        .iter()
        .zip(&data.algorithms)
        .map(|(count, a)| {
            (((count / data.num_instances as f64) * num_cores as f64).floor()
                / a.num_threads as f64)
                .round()
        })
        .collect_vec();
    dbg!(&start_vals);
    for (i, v) in start_vals.iter().enumerate() {
        if v.abs() <= std::f64::EPSILON {
            continue;
        }
        model
            .set_obj_attr(attr::Start, &b[(i, *v as usize - 1)], 1.0)
            .expect("Failed to set initial solution");
    }
    model
        .set_objective(objective_function, ModelSense::Minimize)
        .expect("Failed to set objective function");
    model
        .write("portfolio_model.lp")
        .expect("Failed to write model output file");
    model
        .optimize_with_callback(&mut callback)
        .expect("Error in solution callback");
    let solution = model
        .get_obj_attr_batch(attr::X, b)
        .expect("Model execution failed, no solution");
    let result =
        postprocess_solution(solution, n, num_cores, &data.algorithms);
    dbg!(model.get_attr(attr::ObjVal).unwrap(), m);
    result
}

fn postprocess_solution(
    solution: Vec<f64>,
    n: usize,
    num_cores: usize,
    algorithms: &ndarray::Array1<Algorithm>,
) -> SolverResult {
    let mut portfolio_selection = vec![0.0; n];
    for j in 0..n {
        for k in 0..num_cores {
            portfolio_selection[j] +=
                solution[j * num_cores + k] * (k + 1) as f64;
        }
    }
    let resource_assignments = algorithms
        .iter()
        .zip(portfolio_selection)
        .map(|(algo, cores)| (algo.clone(), cores))
        .collect_vec();
    SolverResult {
        resource_assignments,
    }
}

#[cfg(test)]
mod tests {
    use super::solve;
    use crate::{csv_parser::Data, datastructures::*};

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
    fn test_simple_model() {
        let config = Config {
            files: vec![
                "data/test/algo1.csv".to_string(),
                "data/test/algo2.csv".into(),
            ],
            ..default_config()
        };
        let k = config.num_cores;
        let data = Data::new(config);
        assert_eq!(
            solve(&data, k as usize),
            SolverResult {
                resource_assignments: vec![
                    (
                        Algorithm {
                            algorithm: "algo1".into(),
                            num_threads: 1
                        },
                        0.0
                    ),
                    (
                        Algorithm {
                            algorithm: "algo2".into(),
                            num_threads: 1
                        },
                        2.0
                    ),
                ]
            }
        );
    }

    #[test]
    fn test_seq_vs_par() {
        let config = Config {
            files: vec![
                "data/test/algo1.csv".to_string(),
                "data/test/algo7.csv".into(),
            ],
            num_cores: 8,
            ..default_config()
        };
        let k = config.num_cores;
        let data = Data::new(config);
        assert_eq!(
            solve(&data, k as usize),
            SolverResult {
                resource_assignments: vec![
                    (
                        Algorithm {
                            algorithm: "algo1".into(),
                            num_threads: 1
                        },
                        4.0
                    ),
                    (
                        Algorithm {
                            algorithm: "algo7".into(),
                            num_threads: 4
                        },
                        1.0
                    ),
                ]
            }
        );
    }
}
