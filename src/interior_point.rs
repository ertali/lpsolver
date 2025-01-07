use nalgebra::{DMatrix, DVector};

/// Stores a snapshot of each iteration for the interior point method.
#[derive(Clone, PartialEq)]
pub struct InteriorPointIteration {
    /// D = diag(x)
    pub d_matrix: DMatrix<f64>,
    /// A * D
    pub a_tilde_matrix: DMatrix<f64>,
    /// D * c
    pub c_tilde_vector: DVector<f64>,
    /// P = I - A^T (A A^T)^{-1} A
    pub p_matrix: DMatrix<f64>,
    /// P * c_tilde
    pub cp_vector: DVector<f64>,
    /// The updated x after this iteration
    pub current_x: DVector<f64>,
}

/// If you want a container describing the entire interior point problem:
pub struct InteriorPointProblem {
    /// A matrix (m x n)
    pub a_matrix: DMatrix<f64>,
    /// b vector (m)
    pub b_vector: DVector<f64>,
    /// c vector (n)
    pub c_vector: DVector<f64>,
    /// current x (n), must be strictly > 0 for interior solutions
    pub x_vector: DVector<f64>,
    /// alpha for step size (0 < alpha < 1)
    pub alpha: f64,
    /// Possibly store the constraint types if needed for standardization
    pub constraint_types: Vec<String>,
}

/// A simple error enum for the interior point method
#[derive(Debug)]
pub enum InteriorPointError {
    NoImprovement,
    NotFeasible,
    SingularMatrix(String),
}

/// Convert an LP to standard form if needed (min c'x, Ax=b, x>=0).
/// This is a direct port of your older snippet.
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

    // Copy original coefficients
    new_a.view_mut((0, 0), (m, n)).copy_from(&a);
    // c is row-based in your snippet, so carefully do the copying:
    for i in 0..n {
        new_c[i] = c[i];
    }

    // Add slack or surplus
    let mut slack_idx = n;
    for (i, cons_type) in constraint_types.iter().enumerate() {
        match cons_type.as_str() {
            "<=" => {
                new_a[(i, slack_idx)] = 1.0; // +slack
                slack_idx += 1;
            }
            ">=" => {
                new_a[(i, slack_idx)] = -1.0; // -slack => x>=0
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

/// Creates diag(x), where x is n-dimensional
pub fn create_d_matrix(x: &DVector<f64>) -> DMatrix<f64> {
    let n = x.len();
    let mut d = DMatrix::zeros(n, n);
    for i in 0..n {
        d[(i, i)] = x[i];
    }
    d
}

/// A * D
pub fn calculate_a_tilde(a: &DMatrix<f64>, d: &DMatrix<f64>) -> DMatrix<f64> {
    a * d
}

/// D * c
pub fn calculate_c_tilde(c: &DVector<f64>, d: &DMatrix<f64>) -> DVector<f64> {
    d * c
}

/// P = I - A^T (A A^T)^{-1} A
pub fn calculate_p_matrix(a_tilde: &DMatrix<f64>) -> Result<DMatrix<f64>, InteriorPointError> {
    let n = a_tilde.ncols();
    let i_n = DMatrix::identity(n, n);

    // a_tilde is (m x n), so a_tilde * a_tilde^T => (m x m)
    let a_tilde_t = a_tilde.transpose();
    let mtx = a_tilde * a_tilde_t.clone(); // (m x m)
    let mtx_inv = mtx.try_inverse().ok_or_else(|| {
        InteriorPointError::SingularMatrix("Cannot invert (A*D) * (A*D)^T".to_string())
    })?;

    // Then P = I - A^T * ( (A*D)*(A*D)^T )^{-1} * A
    let p = i_n - a_tilde_t * mtx_inv * a_tilde;
    Ok(p)
}

/// P * c_tilde
pub fn calculate_cp_vector(p: &DMatrix<f64>, c_tilde: &DVector<f64>) -> DVector<f64> {
    p * c_tilde
}

/// The main step of the interior point iteration:
/// x_{k+1} = 1 + alpha * (P * c_tilde) / ||something|| ...
/// or in your older version, you used (alpha / maxval).
pub fn perform_interior_point_iteration(
    problem: &mut InteriorPointProblem,
) -> Result<InteriorPointIteration, InteriorPointError> {
    // 1) Build D = diag(x)
    let d = create_d_matrix(&problem.x_vector);

    // 2) A_tilde = A * D
    let a_tilde = calculate_a_tilde(&problem.a_matrix, &d);

    // 3) c_tilde = D * c
    let c_tilde = calculate_c_tilde(&problem.c_vector, &d);

    // 4) P = I - A^T (A A^T)^{-1} A
    let p = calculate_p_matrix(&a_tilde)?;

    // 5) cp = P * c_tilde
    let cp = calculate_cp_vector(&p, &c_tilde);

    // 6) find the largest absolute value in cp => call it v
    let v = cp.iter().fold(0.0_f64, |acc, &val| acc.max(val.abs()));

    if v < 1e-14 {
        // Means no direction to improve => stuck or optimum
        return Err(InteriorPointError::NoImprovement);
    }

    // 7) update x => x_new = e + (alpha / v)*cp
    // you have e = vector of ones
    let n = problem.x_vector.len();
    let e = DVector::from_element(n, 1.0);
    let factor = problem.alpha / v;
    let new_x = &e + factor * cp.clone();

    // If new_x has negative components => might be an issue in real interior point methods,
    // but for your demonstration, maybe it's okay as is.
    // Overwrite x in the problem so the next iteration sees the updated x
    problem.x_vector = new_x.clone();

    // Return iteration snapshot
    Ok(InteriorPointIteration {
        d_matrix: d,
        a_tilde_matrix: a_tilde,
        c_tilde_vector: c_tilde,
        p_matrix: p,
        cp_vector: cp,
        current_x: new_x,
    })
}

/// A simple "initial feasible point" guess. Real interior point solvers
/// usually do a Phase I or solve an auxiliary problem to find x>=0, Ax=b.
pub fn initial_feasible_point(a: &DMatrix<f64>, b: &DVector<f64>) -> Option<DVector<f64>> {
    // Basic stub: returns a vector of ones.
    // If the dimension doesn't match or if it's obviously infeasible, return None.
    let n = a.ncols();
    if n == 0 {
        return None;
    }
    // You could do a quick check if A*(1 vector) == b, but let's skip for now.
    Some(DVector::from_element(n, 1.0))
}
