use nalgebra::DMatrix;
use std::fmt;

#[derive(Clone, PartialEq)]
pub enum MCoefficient {
    Number(f64),
    M(f64),
}

impl fmt::Display for MCoefficient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MCoefficient::Number(n) => {
                if n.abs() < 1e-10 {
                    write!(f, "0.00")
                } else {
                    write!(f, "{:.2}", n)
                }
            }
            MCoefficient::M(coeff) => {
                if coeff.abs() < 1e-10 {
                    write!(f, "0.00")
                } else if (coeff - 1.0).abs() < 1e-10 {
                    write!(f, "M")
                } else if (coeff + 1.0).abs() < 1e-10 {
                    write!(f, "-M")
                } else {
                    write!(f, "{:.2}M", coeff)
                }
            }
        }
    }
}

#[derive(Debug)]
pub enum SimplexError {
    Infeasible(String),
    Unbounded(String),
    NoSolution(String),
}

pub struct SimplexProblem {
    pub tableau: DMatrix<f64>,
    pub num_vars: usize,
    pub slack_count: usize,
    pub surplus_count: usize,
    pub artificial_count: usize,
    pub constraint_signs: Vec<String>,
    pub rhs: Vec<f64>,
}

pub fn print_tableau(tableau: &DMatrix<f64>) {
    let (rows, cols) = tableau.shape();
    for i in 0..rows {
        for j in 0..cols {
            let val = if tableau[(rows - 1, j)].abs() > 10_000_000.0 - 1e-10 {
                MCoefficient::M(tableau[(i, j)] / 10_000_000.0)
            } else {
                MCoefficient::Number(tableau[(i, j)])
            };
            print!("{} ", val);
        }
        println!();
    }
}

pub fn find_pivot_column(tableau: &DMatrix<f64>) -> Option<usize> {
    let (_, cols) = tableau.shape();
    let last_row = tableau.nrows() - 1;

    let mut min_val = 0.0;
    let mut min_col = None;

    for j in 0..(cols - 1) {
        if tableau[(last_row, j)] < min_val {
            min_val = tableau[(last_row, j)];
            min_col = Some(j);
        }
    }

    min_col
}

pub fn find_pivot_row(tableau: &DMatrix<f64>, pivot_col: usize) -> Option<usize> {
    let (rows, cols) = tableau.shape();
    let mut min_ratio = f64::INFINITY;
    let mut pivot_row = None;

    for i in 0..(rows - 1) {
        if tableau[(i, pivot_col)] > 0.0 {
            let ratio = tableau[(i, cols - 1)] / tableau[(i, pivot_col)];
            if ratio < min_ratio && ratio >= 0.0 {
                min_ratio = ratio;
                pivot_row = Some(i);
            }
        }
    }

    pivot_row
}

pub fn perform_row_operations(tableau: &mut DMatrix<f64>, pivot_row: usize, pivot_col: usize) {
    let (rows, cols) = tableau.shape();
    let pivot_value = tableau[(pivot_row, pivot_col)];

    for j in 0..cols {
        tableau[(pivot_row, j)] /= pivot_value;
    }

    for i in 0..rows {
        if i != pivot_row {
            let factor = tableau[(i, pivot_col)];
            for j in 0..cols {
                tableau[(i, j)] -= factor * tableau[(pivot_row, j)];
            }
        }
    }
}

pub fn is_optimal(tableau: &DMatrix<f64>) -> bool {
    let last_row = tableau.nrows() - 1;
    tableau
        .row(last_row)
        .iter()
        .take(tableau.ncols() - 1)
        .all(|&x| x >= -1e-10)
}

pub fn check_feasibility(tableau: &DMatrix<f64>) -> Result<(), SimplexError> {
    let (rows, cols) = tableau.shape();

    for i in 0..(rows - 1) {
        if tableau[(i, cols - 1)] < -1e-10 {
            return Err(SimplexError::Infeasible(format!(
                "Constraint {} is infeasible: Required value {} is negative",
                i + 1,
                tableau[(i, cols - 1)]
            )));
        }
    }

    Ok(())
}

pub fn check_unboundedness(tableau: &DMatrix<f64>, pivot_col: usize) -> Result<(), SimplexError> {
    let (rows, cols) = tableau.shape();
    let last_row = rows - 1;

    if tableau
        .row(last_row)
        .iter()
        .take(cols - 1)
        .all(|&x| x >= -1e-10)
    {
        return Err(SimplexError::Unbounded(format!(
            "The problem is unbounded along the direction of column {}",
            pivot_col + 1
        )));
    }

    Ok(())
}

pub fn check_no_solution(tableau: &DMatrix<f64>) -> Result<(), SimplexError> {
    let (rows, cols) = tableau.shape();
    let last_row = rows - 1;

    if tableau
        .row(last_row)
        .iter()
        .take(cols - 1)
        .all(|&x| x >= -1e-10)
    {
        return Err(SimplexError::NoSolution(
            "The problem has no solution".to_string(),
        ));
    }

    Ok(())
}

pub fn augment_tableau(
    objective_coeffs: Vec<f64>,
    constraints: Vec<Vec<f64>>,
    constraint_signs: Vec<String>,
    rhs: Vec<f64>,
    artificial_handler: u8,
) -> SimplexProblem {
    let num_vars = objective_coeffs.len();
    let num_constraints = constraints.len();
    let mut slack_count = 0;
    let mut surplus_count = 0;
    let mut artificial_count = 0;

    for (i, sign) in constraint_signs.iter().enumerate() {
        match sign.as_str() {
            "<=" => {
                if rhs[i] <= 0.0 {
                    surplus_count += 1;
                    artificial_count += 1;
                } else {
                    slack_count += 1;
                }
            }
            ">=" => {
                surplus_count += 1;
                artificial_count += 1;
            }
            "=" => artificial_count += 1,
            _ => panic!("Invalid constraint sign"),
        }
    }

    let total_cols = num_vars + slack_count + surplus_count + artificial_count + 1;
    let total_rows = num_constraints + 1;
    let mut tableau = DMatrix::zeros(total_rows, total_cols);

    for (j, &coeff) in objective_coeffs.iter().enumerate() {
        tableau[(total_rows - 1, j)] = -coeff;
    }

    let mut slack_idx = num_vars;
    let mut surplus_idx = slack_idx + slack_count;
    let mut artificial_idx = surplus_idx + surplus_count;
    let big_m = 10_000_000.0;

    match artificial_handler {
        0 => {
            for (i, (constraint, sign)) in constraints.iter().zip(&constraint_signs).enumerate() {
                for (j, &coeff) in constraint.iter().enumerate() {
                    tableau[(i, j)] = coeff;
                }

                match sign.as_str() {
                    "<=" => {
                        tableau[(i, slack_idx)] = 1.0;
                        slack_idx += 1;
                    }
                    ">=" => {
                        tableau[(i, surplus_idx)] = -1.0;
                        tableau[(i, artificial_idx)] = 1.0;
                        tableau[(total_rows - 1, artificial_idx)] = big_m;
                        for j in 0..total_cols {
                            tableau[(total_rows - 1, i)] -= big_m * tableau[(i, j)];
                        }
                        surplus_idx += 1;
                        artificial_idx += 1;
                    }
                    "=" => {
                        tableau[(i, artificial_idx)] = 1.0;
                        artificial_idx += 1;
                    }
                    _ => panic!("Invalid constraint sign"),
                }

                tableau[(i, total_cols - 1)] = rhs[i];
            }
            SimplexProblem {
                tableau,
                num_vars,
                slack_count,
                surplus_count,
                artificial_count,
                constraint_signs,
                rhs,
            }
        }
        1 => {
            let mut phase_one_tableau: DMatrix<f64> = DMatrix::zeros(total_rows + 1, total_cols);

            for i in 0..total_rows {
                for j in 0..total_cols {
                    phase_one_tableau[(i, j)] = tableau[(i, j)];
                }
            }

            let artificial_start = num_vars + slack_count + surplus_count;
            for j in artificial_start..(total_cols - 1) {
                phase_one_tableau[(total_rows, j)] = 1.0;
            }

            for i in 0..num_constraints {
                if constraint_signs[i] == ">" || constraint_signs[i] == "=" {
                    for j in 0..total_cols {
                        phase_one_tableau[(total_rows, j)] -= phase_one_tableau[(i, j)];
                    }
                }
            }

            SimplexProblem {
                tableau: phase_one_tableau,
                num_vars,
                slack_count,
                surplus_count,
                artificial_count,
                constraint_signs,
                rhs,
            }
        }
        _ => {
            panic!("Unsupported artificial handler: {}", artificial_handler);
        }
    }
}

pub fn phase_one_solver(problem: &mut SimplexProblem) -> Result<DMatrix<f64>, SimplexError> {
    let original_rows = problem.tableau.nrows() - 1;
    let artificial_start = problem.num_vars + problem.slack_count + problem.surplus_count;
    let phase_one_row = problem.tableau.ncols() - 1;

    check_feasibility(&problem.tableau)?;

    while !is_optimal(&problem.tableau) {
        if let Some(pivot_col) = find_pivot_column(&problem.tableau) {
            if let Some(pivot_row) = find_pivot_row(&problem.tableau, pivot_col) {
                perform_row_operations(&mut problem.tableau, pivot_row, pivot_col);
            } else {
                return Err(SimplexError::Unbounded(
                    "No valid pivot row in Phase I. Artificial variable constraints can't be satisfied".to_string(),
                ));
            }
        } else {
            break;
        }
    }

    if problem.tableau[(phase_one_row, problem.tableau.ncols() - 1)].abs() > 1e-10 {
        return Err(SimplexError::NoSolution(
            "The problem has no feasible solution".to_string(),
        ));
    }

    let mut phase_two_tableau = DMatrix::zeros(original_rows, artificial_start + 1);
    for i in 0..original_rows {
        for j in 0..artificial_start {
            phase_two_tableau[(i, j)] = problem.tableau[(i, j)];
        }
        phase_two_tableau[(i, artificial_start)] =
            problem.tableau[(i, problem.tableau.ncols() - 1)];
    }

    Ok(phase_two_tableau)
}

pub fn extract_solution(tableau: &DMatrix<f64>) -> Vec<f64> {
    let (rows, cols) = tableau.shape();
    let mut solution = vec![0.0; cols - 1];

    for j in 0..(cols - 1) {
        let mut basic_var = None;
        let mut basic_val = 0.0;

        for i in 0..(rows - 1) {
            if tableau[(i, j)] == 1.0
                && (0..rows)
                    .filter(|&k| k != i)
                    .all(|k| tableau[(k, j)] == 0.0)
            {
                basic_var = Some(j);
                basic_val = tableau[(i, cols - 1)];
                break;
            }
        }

        if let Some(j) = basic_var {
            solution[j] = basic_val;
        }
    }

    solution
}
