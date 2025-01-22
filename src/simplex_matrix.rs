use nalgebra::{DMatrix, DVector};

#[derive(Clone, PartialEq)]
pub struct SimplexMatrixIteration {
    pub a_matrix: DMatrix<f64>,
    pub b_vector: DVector<f64>,
    pub c_vector: DVector<f64>,
    pub z_value: f64,
    pub x_vector: DVector<f64>,
    pub x_s_vector: DVector<f64>,
    pub cp_b_inverse_a: DMatrix<f64>,
    pub cp_b_inverse: DMatrix<f64>,
    pub b_inverse_a: DMatrix<f64>,
    pub b_inverse: DMatrix<f64>,
}

#[derive(Debug)]
pub enum MatrixSimplexError {
    Infeasible(String),
    Unbounded(String),
    NoSolution(String),
    InversionFailed(String),
}

pub struct SimplexMatrixProblem {
    pub a_matrix: DMatrix<f64>,
    pub b_vector: DVector<f64>,
    pub c_vector: DVector<f64>,
    pub x_vector: DVector<f64>,
    pub b_inverse: DMatrix<f64>,
    pub c_b_vector: DVector<f64>,
    pub basis_indices: Vec<usize>,
}

pub fn extract_matrices(
    tableau: &DMatrix<f64>,
    num_variables: usize,
    num_constraints: usize,
) -> (DMatrix<f64>, DVector<f64>, DVector<f64>) {
    let (rows, cols) = tableau.shape();

    let a = tableau
        .view((0, 0), (num_constraints, num_variables))
        .into_owned();

    let b = DVector::from_iterator(
        num_constraints,
        (0..num_constraints).map(|i| tableau[(i, cols - 1)]),
    );

    let c = DVector::from_iterator(
        num_variables,
        (0..num_variables).map(|j| -tableau[(rows - 1, j)]),
    );

    (a, b, c)
}

fn calculate_inverse(matrix: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    matrix.clone().try_inverse()
}

pub fn calculate_matrix_iteration(
    problem: &SimplexMatrixProblem,
) -> Result<SimplexMatrixIteration, MatrixSimplexError> {
    let a = &problem.a_matrix;
    let b = &problem.b_vector;
    let c = &problem.c_vector;
    let b_inverse = &problem.b_inverse;
    let c_b = &problem.c_b_vector;
    let x_vector = &problem.x_vector;

    let b_inverse_a = b_inverse * a;

    let cp_b_inverse = (c_b.transpose() * b_inverse).transpose(); // (m x 1)

    let cp_b_inverse_a = (&cp_b_inverse * a - c.transpose()).transpose(); // (n x 1)

    let z_value = c.dot(x_vector);

    let x_s_vector = DVector::zeros(b.len());

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

pub fn perform_simplex_matrix_iteration(
    problem: &mut SimplexMatrixProblem,
) -> Result<bool, MatrixSimplexError> {
    let m = problem.b_vector.len();
    let n = problem.a_matrix.ncols();

    let iteration_data = calculate_matrix_iteration(problem)?;
    let reduced_costs = &iteration_data.cp_b_inverse_a;

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
        return Ok(false);
    }
    let entering_var_index = entering_col.unwrap();

    let b_inv_a_enter = &problem.b_inverse * problem.a_matrix.column(entering_var_index);
    let mut leaving_row = None;
    let mut min_ratio = f64::INFINITY;

    for i in 0..m {
        let coeff = b_inv_a_enter[i];
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
            return Err(MatrixSimplexError::Unbounded(
                "No valid pivot row found. The problem is unbounded.".to_string(),
            ));
        }
    };

    let leaving_var_index = problem.basis_indices[leaving_row_idx];
    problem.basis_indices[leaving_row_idx] = entering_var_index;

    let new_b = extract_basis_matrix(&problem.a_matrix, &problem.basis_indices);
    let new_b_inverse = calculate_inverse(&new_b).ok_or_else(|| {
        MatrixSimplexError::InversionFailed("Failed to invert new B matrix!".to_string())
    })?;

    problem.b_inverse = new_b_inverse;

    let new_c_b = extract_basis_coefficients(&problem.c_vector, &problem.basis_indices);
    problem.c_b_vector = new_c_b;

    let x_b = &problem.b_inverse * &problem.b_vector;
    for i in 0..problem.x_vector.len() {
        problem.x_vector[i] = 0.0;
    }
    for (row_i, &col_j) in problem.basis_indices.iter().enumerate() {
        problem.x_vector[col_j] = x_b[row_i];
    }

    Ok(true)
}

fn extract_basis_matrix(a: &DMatrix<f64>, basis_indices: &[usize]) -> DMatrix<f64> {
    let m = a.nrows();
    let mut b = DMatrix::zeros(m, m);
    for (col, &idx) in basis_indices.iter().enumerate() {
        b.set_column(col, &a.column(idx));
    }
    b
}

fn extract_basis_coefficients(c: &DVector<f64>, basis_indices: &[usize]) -> DVector<f64> {
    DVector::from_iterator(basis_indices.len(), basis_indices.iter().map(|&idx| c[idx]))
}
