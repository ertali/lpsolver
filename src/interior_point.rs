use nalgebra::{DMatrix, DVector};

#[derive(Clone, PartialEq)]
pub struct InteriorPointIteration {
    pub d_matrix: DMatrix<f64>,
    pub a_tilde_matrix: DMatrix<f64>,
    pub c_tilde_vector: DVector<f64>,
    pub p_matrix: DMatrix<f64>,
    pub cp_vector: DVector<f64>,
    pub current_x: DVector<f64>,
}

pub struct InteriorPointProblem {
    pub a_matrix: DMatrix<f64>,
    pub b_vector: DVector<f64>,
    pub c_vector: DVector<f64>,
    pub x_vector: DVector<f64>,
    pub alpha: f64,
    pub constraint_types: Vec<String>,
}

#[derive(Debug)]
pub enum InteriorPointError {
    NoImprovement,
    NotFeasible,
    SingularMatrix(String),
}

pub fn standardize_problem(
    a: DMatrix<f64>,
    b: DVector<f64>,
    c: DVector<f64>,
    constraint_types: Vec<String>,
) -> (DMatrix<f64>, DVector<f64>, DVector<f64>) {
    let (m, n) = a.shape();
    let mut slack_count = 0;

    for cons_type in &constraint_types {
        if cons_type != "=" {
            slack_count += 1;
        }
    }

    let total_vars = n + slack_count;
    let mut new_a = DMatrix::zeros(m, total_vars);
    let mut new_c = DVector::zeros(total_vars);

    new_a.view_mut((0, 0), (m, n)).copy_from(&a);
    for i in 0..n {
        new_c[i] = c[i];
    }

    let mut slack_idx = n;
    for (i, cons_type) in constraint_types.iter().enumerate() {
        match cons_type.as_str() {
            "<=" => {
                new_a[(i, slack_idx)] = 1.0;
                slack_idx += 1;
            }
            ">=" => {
                new_a[(i, slack_idx)] = -1.0;
                slack_idx += 1;
            }
            "=" => {
                // no slack
            }
            _ => panic!("Invalid constraint type: {}", cons_type),
        }
    }

    (new_a, b, new_c)
}

pub fn create_d_matrix(x: &DVector<f64>) -> DMatrix<f64> {
    let n = x.len();
    let mut d = DMatrix::zeros(n, n);
    for i in 0..n {
        d[(i, i)] = x[i];
    }
    d
}

pub fn calculate_a_tilde(a: &DMatrix<f64>, d: &DMatrix<f64>) -> DMatrix<f64> {
    a * d
}

pub fn calculate_c_tilde(c: &DVector<f64>, d: &DMatrix<f64>) -> DVector<f64> {
    d * c
}

pub fn calculate_p_matrix(a_tilde: &DMatrix<f64>) -> Result<DMatrix<f64>, InteriorPointError> {
    let n = a_tilde.ncols();
    let i_n = DMatrix::identity(n, n);

    let a_tilde_t = a_tilde.transpose();
    let mtx = a_tilde * a_tilde_t.clone(); // (m x m)
    let mtx_inv = mtx.try_inverse().ok_or_else(|| {
        InteriorPointError::SingularMatrix("Cannot invert (A*D) * (A*D)^T".to_string())
    })?;

    let p = i_n - a_tilde_t * mtx_inv * a_tilde;
    Ok(p)
}

pub fn calculate_cp_vector(p: &DMatrix<f64>, c_tilde: &DVector<f64>) -> DVector<f64> {
    p * c_tilde
}

pub fn perform_interior_point_iteration(
    problem: &mut InteriorPointProblem,
) -> Result<InteriorPointIteration, InteriorPointError> {
    let d = create_d_matrix(&problem.x_vector);

    let a_tilde = calculate_a_tilde(&problem.a_matrix, &d);

    let c_tilde = calculate_c_tilde(&problem.c_vector, &d);

    let p = calculate_p_matrix(&a_tilde)?;

    let cp = calculate_cp_vector(&p, &c_tilde);

    let v = cp.iter().fold(0.0_f64, |acc, &val| acc.max(val.abs()));

    if v < 1e-14 {
        return Err(InteriorPointError::NoImprovement);
    }

    let n = problem.x_vector.len();
    let e = DVector::from_element(n, 1.0);
    let factor = problem.alpha / v;
    let new_x = &e + factor * cp.clone();

    problem.x_vector = new_x.clone();

    Ok(InteriorPointIteration {
        d_matrix: d,
        a_tilde_matrix: a_tilde,
        c_tilde_vector: c_tilde,
        p_matrix: p,
        cp_vector: cp,
        current_x: new_x,
    })
}

pub fn initial_feasible_point(a: &DMatrix<f64>, b: &DVector<f64>) -> Option<DVector<f64>> {
    let n = a.ncols();
    if n == 0 {
        return None;
    }
    Some(DVector::from_element(n, 1.0))
}
