use nalgebra::{DMatrix, DVector};

/// An iteration snapshot for the matrix-based Simplex method.
#[derive(Clone, PartialEq)]
pub struct SimplexMatrixIteration {
    pub a_matrix: DMatrix<f64>,
    pub b_vector: DVector<f64>,
    pub c_vector: DVector<f64>,
    pub z_value: f64,
    /// The full variable solution (including nonbasic = 0)
    pub x_vector: DVector<f64>,
    /// Slack/surplus or other extra variables (not always used)
    pub x_s_vector: DVector<f64>,
    /// cB * B^-1 * A - c, i.e. the reduced costs row (as a 1 x n matrix)
    pub cp_b_inverse_a: DMatrix<f64>,
    /// cB * B^-1 (as a 1 x m matrix)
    pub cp_b_inverse: DMatrix<f64>,
    /// B^-1 * A (as an m x n matrix)
    pub b_inverse_a: DMatrix<f64>,
    /// Current inverse of the basis
    pub b_inverse: DMatrix<f64>,
}

/// Errors specific to the matrix-based Simplex approach.
#[derive(Debug)]
pub enum MatrixSimplexError {
    Infeasible(String),
    Unbounded(String),
    NoSolution(String),
    InversionFailed(String),
}

/// A container for everything we need to do matrix-form iterations.
/// (This is analogous to the `SimplexProblem` in `simplex.rs`, but specialized.)
pub struct SimplexMatrixProblem {
    /// Constraint matrix A (m x n)
    pub a_matrix: DMatrix<f64>,
    /// RHS vector b (m x 1)
    pub b_vector: DVector<f64>,
    /// Objective coefficients c (n x 1)
    pub c_vector: DVector<f64>,
    /// Current solution x (n x 1)
    pub x_vector: DVector<f64>,
    /// Current basis inverse B^-1 (m x m)
    pub b_inverse: DMatrix<f64>,
    /// cB vector (coefficients for the basic variables)
    pub c_b_vector: DVector<f64>,
    /// Indices of the columns in A that form the current basis (length = m)
    pub basis_indices: Vec<usize>,
}

/// Extract A, b, c from a tableau in case you need to initialize a
/// matrix-form problem from a simplex tableau. (Optional)
pub fn extract_matrices(
    tableau: &DMatrix<f64>,
    num_variables: usize,
    num_constraints: usize,
) -> (DMatrix<f64>, DVector<f64>, DVector<f64>) {
    let (rows, cols) = tableau.shape();

    // A matrix (first m rows, first n cols)
    let a = tableau
        .view((0, 0), (num_constraints, num_variables))
        .into_owned();

    // b vector (right-most column of the first m rows)
    let b = DVector::from_iterator(
        num_constraints,
        (0..num_constraints).map(|i| tableau[(i, cols - 1)]),
    );

    // c vector (negative of last row’s first n columns => because in tableau we store -c in the bottom row)
    let c = DVector::from_iterator(
        num_variables,
        (0..num_variables).map(|j| -tableau[(rows - 1, j)]),
    );

    (a, b, c)
}

/// Attempt to invert a matrix. Returns None if singular.
fn calculate_inverse(matrix: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    matrix.clone().try_inverse()
}

/// Creates a `SimplexMatrixIteration` snapshot, which can help debug/inspect each step.
pub fn calculate_matrix_iteration(
    problem: &SimplexMatrixProblem,
) -> Result<SimplexMatrixIteration, MatrixSimplexError> {
    let a = &problem.a_matrix;
    let b = &problem.b_vector;
    let c = &problem.c_vector;
    let b_inverse = &problem.b_inverse;
    let c_b = &problem.c_b_vector;
    let x_vector = &problem.x_vector;

    // B^-1 * A
    let b_inverse_a = b_inverse * a;

    // cB^T * B^-1
    let cp_b_inverse = (c_b.transpose() * b_inverse).transpose(); // (m x 1)

    // cB^T * B^-1 * A - c^T => shape is 1 x n, then we transpose to get n x 1
    let cp_b_inverse_a = (&cp_b_inverse * a - c.transpose()).transpose(); // (n x 1)

    // Current objective value: z = c^T x
    let z_value = c.dot(x_vector);

    // Slack/surplus vector is optional, but we can set to zeros or something if you prefer
    let x_s_vector = DVector::zeros(b.len());

    // Build an iteration snapshot
    Ok(SimplexMatrixIteration {
        a_matrix: a.clone(),
        b_vector: b.clone(),
        c_vector: c.clone(),
        z_value,
        x_vector: x_vector.clone(),
        x_s_vector,
        cp_b_inverse_a: DMatrix::from_row_slice(1, cp_b_inverse_a.len(), cp_b_inverse_a.as_slice()),
        cp_b_inverse: DMatrix::from_row_slice(1, cp_b_inverse.len(), cp_b_inverse.as_slice()),
        b_inverse_a,
        b_inverse: b_inverse.clone(),
    })
}

/// Perform one iteration of the matrix-form Simplex method.
/// This function updates the `problem` in place:
///  1. Finds entering variable (most negative reduced cost).
///  2. Finds leaving variable (minimum ratio test).
///  3. Updates basis, re-computes B^-1 and x.
/// Returns `Ok(true)` if a pivot was done, `Ok(false)` if already optimal, or an error if unbounded/infeasible/etc.
pub fn perform_simplex_matrix_iteration(
    problem: &mut SimplexMatrixProblem,
) -> Result<bool, MatrixSimplexError> {
    let m = problem.b_vector.len();
    let n = problem.a_matrix.ncols();

    // 1. Compute reduced costs: r_j = c_j - c_B^T * B^-1 * A_j
    // We can reuse `calculate_matrix_iteration` to get cp_b_inverse_a.
    let iteration_data = calculate_matrix_iteration(problem)?;
    // cp_b_inverse_a is shape (n x 1), each entry is the reduced cost for column j
    let reduced_costs = &iteration_data.cp_b_inverse_a;

    // 2. Check optimality: if all reduced costs >= 0 (assuming a max problem),
    //    we're done. (If Minimization, we'd do <= 0 check, or else invert logic.)
    let mut entering_col = None;
    let mut min_val = 0.0;
    for j in 0..n {
        let cost_j = reduced_costs[(j, 0)];
        if cost_j < min_val {
            min_val = cost_j;
            entering_col = Some(j);
        }
    }
    if entering_col.is_none() {
        // No negative reduced cost => optimal
        return Ok(false);
    }
    let entering_var_index = entering_col.unwrap();

    // 3. Determine leaving variable via ratio test
    //    We compute direction d = B^-1 * A_{entering}
    let b_inv_a_enter = &problem.b_inverse * problem.a_matrix.column(entering_var_index);
    let mut leaving_row = None;
    let mut min_ratio = f64::INFINITY;

    for i in 0..m {
        let coeff = b_inv_a_enter[i];
        // Only consider positive directions
        if coeff > 1e-14 {
            let ratio = problem.x_vector[problem.basis_indices[i]] / coeff;
            if ratio < min_ratio {
                min_ratio = ratio;
                leaving_row = Some(i);
            }
        }
    }
    let leaving_row_idx = match leaving_row {
        Some(r) => r,
        None => {
            // If no positive entries => unbounded
            return Err(MatrixSimplexError::Unbounded(
                "No valid pivot row found. The problem is unbounded.".to_string(),
            ));
        }
    };

    // 4. Update basis_indices: basis_indices[leaving_row_idx] = entering_var_index
    let leaving_var_index = problem.basis_indices[leaving_row_idx];
    problem.basis_indices[leaving_row_idx] = entering_var_index;

    // 5. Recompute B^-1 and c_B
    //    Re-build B from the new basis, invert it, compute new c_B, etc.
    let new_b = extract_basis_matrix(&problem.a_matrix, &problem.basis_indices);
    let new_b_inverse = calculate_inverse(&new_b).ok_or_else(|| {
        MatrixSimplexError::InversionFailed("Failed to invert new B matrix!".to_string())
    })?;

    // Update B^-1
    problem.b_inverse = new_b_inverse;

    // Update c_B
    let new_c_b = extract_basis_coefficients(&problem.c_vector, &problem.basis_indices);
    problem.c_b_vector = new_c_b;

    // 6. Update x_vector by computing x_B = B^-1 * b, then place those values in the correct positions
    let x_b = &problem.b_inverse * &problem.b_vector;
    for i in 0..problem.x_vector.len() {
        problem.x_vector[i] = 0.0;
    }
    for (row_i, &col_j) in problem.basis_indices.iter().enumerate() {
        problem.x_vector[col_j] = x_b[row_i];
    }

    Ok(true)
}

/// Build the basis matrix from A given the basis indices
fn extract_basis_matrix(a: &DMatrix<f64>, basis_indices: &[usize]) -> DMatrix<f64> {
    let m = a.nrows();
    let mut b = DMatrix::zeros(m, m);
    for (col, &idx) in basis_indices.iter().enumerate() {
        b.set_column(col, &a.column(idx));
    }
    b
}

/// Extract c_B from c by picking out the coefficients for the basis columns
fn extract_basis_coefficients(c: &DVector<f64>, basis_indices: &[usize]) -> DVector<f64> {
    DVector::from_iterator(basis_indices.len(), basis_indices.iter().map(|&idx| c[idx]))
}
