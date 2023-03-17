use std::cmp::Ordering;

use crate::datastructures::*;
use itertools::Itertools;

use crate::csv_parser::Data;
use grb::prelude::*;
use ndarray::{Array1, Array2, Array3};

pub fn solve(
    data: &Data,
    num_cores: usize,
    timeout: Timeout,
) -> OptimizationResult {
    let mut model =
        Model::new("portfolio_model").expect("Failed to create Gurobi Model");
    model.set_param(param::NumericFocus, 1).unwrap();
    // model.set_param(param::SolFiles, "portfolio_model".to_string())?;
    model.set_param(param::TimeLimit, timeout.0).unwrap();
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
            let opt = (obj / obj_bnd).abs() < f64::EPSILON;
            let res = postprocess_solution(
                sol,
                n,
                num_cores,
                &data.algorithms,
                "intermediate_portfolio",
                opt,
            );
            println!("{res}");
            dbg!(obj_bnd, obj);
        }
        Ok(())
    };

    let mut initial_solution = vec![0.0; n * num_cores];
    for (i, v) in get_b_start(
        &data.best_per_instance_count,
        &data.algorithms,
        m,
        num_cores,
    )
    .iter()
    .enumerate()
    {
        if v.abs() <= std::f64::EPSILON {
            continue;
        }
        model
            .set_obj_attr(attr::Start, &b[(i, *v as usize - 1)], 1.0)
            .expect("Failed to set initial solution");
        initial_solution[i * num_cores + *v as usize - 1] = 1.0;
    }

    let initial_portfolio = postprocess_solution(
        initial_solution,
        n,
        num_cores,
        &data.algorithms,
        "initial_portfolio",
        false,
    );
    dbg!(&initial_portfolio);
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
    let gap = model.get_attr(attr::MIPGap).unwrap_or(f64::MAX);
    let final_portfolio = postprocess_solution(
        solution,
        n,
        num_cores,
        &data.algorithms,
        "final_portfolio",
        gap.abs() < f64::EPSILON,
    );
    dbg!(model.get_attr(attr::ObjVal).unwrap(), m);
    OptimizationResult {
        initial_portfolio,
        final_portfolio,
        gap,
    }
}

fn postprocess_solution(
    solution: Vec<f64>,
    n: usize,
    num_cores: usize,
    algorithms: &ndarray::Array1<Algorithm>,
    portfolio_name: &str,
    opt: bool,
) -> Portfolio {
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
    let name = if opt {
        [portfolio_name, "opt"].join("_")
    } else {
        portfolio_name.to_string()
    };
    Portfolio {
        name,
        resource_assignments,
    }
}

fn get_b_start(
    counts: &ndarray::Array1<f64>,
    algorithms: &ndarray::Array1<Algorithm>,
    m: usize,
    num_cores: usize,
) -> Vec<f64> {
    let fractions = counts
        .iter()
        .zip(algorithms)
        .map(|(count, a)| {
            (count / m as f64) * (num_cores as f64 / a.num_threads as f64)
        })
        .collect_vec();
    let steps = algorithms.iter().map(|a| a.num_threads).collect_vec();
    round_to_sum(&fractions, &steps, num_cores as u32)
}

fn round_to_sum(fractions: &[f64], steps: &Vec<u32>, sum: u32) -> Vec<f64> {
    let (mut rounded, mut losses): (Vec<f64>, Vec<f64>) =
        fractions.iter().map(|f| (f.floor(), f - f.floor())).unzip();
    let mut remainder = sum
        - rounded
            .iter()
            .zip(steps)
            .fold(0, |acc, (v, s)| acc + *v as u32 * s);
    while remainder > 0 {
        if let Some(highest_loss_idx) = losses
            .iter()
            .zip(steps)
            .position_max_by(|(l1, &s1), (l2, &s2)| {
                if s1 <= remainder && s2 <= remainder {
                    l1.partial_cmp(l2).unwrap()
                } else if s1 > remainder {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            })
        {
            remainder -= steps[highest_loss_idx];
            losses[highest_loss_idx] = 0.0;
            rounded[highest_loss_idx] += 1.0;
        } else {
            break;
        }
    }
    assert_eq!(
        rounded
            .iter()
            .zip(steps)
            .fold(0, |acc, (v, s)| acc + *v as u32 * s),
        sum
    );
    rounded
}

#[cfg(test)]
mod tests;
