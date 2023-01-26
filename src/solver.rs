use itertools::Itertools;

use crate::csv_parser::Data;
use anyhow::Result;
use grb::prelude::*;
use ndarray::{Array1, Array2, Array3};

pub fn solve(data: &Data, num_cores: usize) -> Result<Vec<f64>> {
    let mut model = Model::new("portfolio_model")?;
    model.set_param(param::NumericFocus, 1)?;
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
    let c_1 = a
        .indexed_iter()
        .map(|((i, j, k), &val_a)| {
            model.add_constr(
                format!("c1_{i}_{j}_{k}").as_str(),
                c!(val_a * e_min[(i, j, k)] <= q[i]),
            )
        })
        .collect_vec();

    // constraint 2
    let c_2 = b
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
        .map(|row| {
            row.into_iter()
                .zip(1..=num_cores)
                .map(|(var, k)| *var * k)
                .grb_sum()
        })
        .grb_sum();
    let c_3 = model.add_constr("c3", c!(sums == num_cores));

    // constraint 4
    let c_4 = a
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
    let c_5 = a
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

    model.set_objective(objective_function, ModelSense::Minimize)?;
    model.write("portfolio_model.lp")?;
    model.optimize()?;
    let repeats_assignment =
        model.get_obj_attr_batch(attr::X, b)?;
    let mut portfolio_selection = vec![0.0; num_cores];
    for j in 0..n {
        for k in 0..num_cores {
            portfolio_selection[j] += repeats_assignment[j * n + k] * (k + 1) as f64;
        }
    }
    Ok(portfolio_selection)
}

#[cfg(test)]
mod tests {
    use crate::csv_parser::Data;
    use super::solve;

    #[test]
    fn test_simple_model() {
        let csv_paths = vec!["data/test/algo1.csv", "data/test/algo2.csv"];
        let num_cores = 2;
        let data = Data::new(&csv_paths, num_cores)
            .expect("Error while reading data");
        assert_eq!(solve(&data, num_cores as usize).unwrap(), vec![0.0, 2.0]);
    }
}
