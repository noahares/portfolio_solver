use itertools::Itertools;

use anyhow::Result;
use grb::{*, prelude::GurobiSum};
use ndarray::{Array3, Array2, Array1};

use crate::datastructures::*;
pub fn solve(dataframe: &Dataframe, num_cores: usize) -> Result<Array1::<grb::Var>> {
    let mut model = Model::new("portfolio_model")?;
    let (n, m) = (dataframe.algorithms.len(), dataframe.instances.len());

    let a = Array3::<grb::Var>::from_elem((m, n, num_cores), add_binvar!(model)?);
    let b = Array2::<grb::Var>::from_elem((n, num_cores), add_binvar!(model)?);
    let assigned_resources_per_algo = Array1::<grb::Var>::from_elem(n, add_ctsvar!(model, bounds: 0..num_cores)?);
    let q = Array1::<grb::Var>::from_elem(n, add_ctsvar!(model, bounds: ..)?);
    let best_per_instance = Array1::<f64>::ones(n);

    let e_min = Array3::<f64>::zeros([m, n, num_cores]);

    // constraint 1
    let c_1 = a.indexed_iter()
        .map(|((i, j, k), &val_a)| model.add_constr(format!("c1_{i}_{j}_{k}").as_str(), c!(val_a * e_min[(i, j, k)] <= q[i])))
        .collect_vec();

    // constraint 2
    let c_2 = b.rows()
        .into_iter().enumerate()
        .map(|(i, row)| model.add_constr(format!("c2_{i}").as_str(), c!(row.into_iter().grb_sum() <= 1)))
        .collect_vec();

    // constraint 3
    let sums = b.rows()
        .into_iter()
        .map(|row| row.into_iter().zip(assigned_resources_per_algo.iter()).map(|(var, k)| *var * *k).grb_sum())
        .grb_sum();
    let c_3 = model.add_constr("c3", c!(sums == num_cores));

    // constraint 4
    let c_4 = a.rows()
        .into_iter().enumerate()
        .map(|(i, row)| model.add_constr(format!("c4_{i}").as_str(), c!(row.iter().grb_sum() == 1)))
        .collect_vec();

    // constraint 5
    let c_5 = a.indexed_iter()
        .map(|((i, j, k), &val_a)| model.add_constr(format!("c5_{i}_{j}_{k}").as_str(), c!(val_a <= b[(j, k)])))
        .collect_vec();

    let objective_function = q.iter().zip(best_per_instance.iter())
        .map(|(&var, &best)| var * (1.0 / best))
        .grb_sum();

    model.set_objective(objective_function, ModelSense::Minimize)?;
    model.write("portfolio_model.lp")?;
    model.optimize()?;
    Ok(assigned_resources_per_algo)
}
