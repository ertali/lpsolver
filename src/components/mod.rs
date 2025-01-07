use crate::interior_point::{
    initial_feasible_point, perform_interior_point_iteration, standardize_problem,
    InteriorPointError, InteriorPointIteration, InteriorPointProblem,
};
use crate::simplex::*;
use crate::simplex_matrix::*;
use input_form::*;
use interior_point_view::InteriorPointView;
use log;
use nalgebra::{DMatrix, DVector};
use simplex_matrix_view::SimplexMatrixView;
use tableau_view::TableauView;
use yew::prelude::*;

mod input_form;
mod interior_point_view;
mod simplex_matrix_view;
mod tableau_view;

#[derive(PartialEq, Clone)]
pub enum SolverType {
    None,
    SimplexTableau,
    SimplexMatrix,
    InteriorPoint,
    SimplexGrapher,
}

pub struct App {
    solver_type: SolverType,
    iterations: Vec<DMatrix<f64>>,
    matrix_iterations: Vec<SimplexMatrixIteration>,
    interior_iterations: Vec<InteriorPointIteration>,
    problem_size: Option<(usize, usize)>,
    step: usize,
    maximization: bool,
    solver_data: SolverData,
}

pub enum SolverData {
    None,
    SimplexTableau {
        tableau: DMatrix<f64>,
        method: String, // Two-Phase or Big M
    },
    SimplexMatrix {
        a_matrix: DMatrix<f64>,
        b_vector: DVector<f64>,
        c_vector: DVector<f64>,
        x_vector: DVector<f64>,
        b_inverse: DMatrix<f64>,
        c_b_vector: DVector<f64>,
        basis_indices: Vec<usize>,
    },
    InteriorPoint {
        // "original" problem data
        a_matrix: DMatrix<f64>,
        b_vector: DVector<f64>,
        c_vector: DVector<f64>,

        // iteration snapshot fields
        d_matrix: DMatrix<f64>,
        a_tilde_matrix: DMatrix<f64>,
        c_tilde_vector: DVector<f64>,
        p_matrix: DMatrix<f64>,
        cp_vector: DVector<f64>,
        current_x: DVector<f64>,

        alpha: f64,

        initial_point: Vec<f64>,
    },
}

pub enum Msg {
    SelectSolver(SolverType),
    SetProblemSize(usize, usize),
    StartSimplexTableau(DMatrix<f64>, bool, String),
    StartSimplexMatrix(DMatrix<f64>, DVector<f64>, DVector<f64>),
    StartInteriorPoint(DMatrix<f64>, DVector<f64>, DVector<f64>, f64, Vec<f64>),
    SetInitialPoint(DVector<f64>),
    NextStep,
    Reset,
    BackToSelection,
}

impl Component for App {
    type Message = Msg;
    type Properties = ();

    fn create(_ctx: &Context<Self>) -> Self {
        Self {
            solver_type: SolverType::None,
            iterations: vec![],
            matrix_iterations: vec![],
            interior_iterations: vec![],
            problem_size: None,
            step: 0,
            maximization: true,
            solver_data: SolverData::None,
        }
    }

    fn update(&mut self, _ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::SelectSolver(solver_type) => {
                self.solver_type = solver_type;
                self.solver_data = match &self.solver_type {
                    SolverType::SimplexTableau => SolverData::SimplexTableau {
                        tableau: DMatrix::zeros(0, 0),
                        method: String::from("Big-M"),
                    },

                    SolverType::SimplexMatrix => SolverData::SimplexMatrix {
                        a_matrix: DMatrix::zeros(0, 0),
                        b_vector: DVector::zeros(0),
                        c_vector: DVector::zeros(0),
                        x_vector: DVector::zeros(0),
                        b_inverse: DMatrix::zeros(0, 0),
                        c_b_vector: DVector::zeros(0),
                        basis_indices: Vec::new(),
                    },

                    SolverType::InteriorPoint => SolverData::InteriorPoint {
                        a_matrix: DMatrix::zeros(0, 0),
                        b_vector: DVector::zeros(0),
                        c_vector: DVector::zeros(0),

                        d_matrix: DMatrix::zeros(0, 0),
                        a_tilde_matrix: DMatrix::zeros(0, 0),
                        c_tilde_vector: DVector::zeros(0),
                        p_matrix: DMatrix::zeros(0, 0),
                        cp_vector: DVector::zeros(0),
                        current_x: DVector::zeros(0),
                        alpha: 0.5,
                        initial_point: vec![],
                    },

                    _ => SolverData::None,
                };
                self.iterations.clear();
                self.step = 0;
                true
            }

            Msg::SetProblemSize(vars, cons) => {
                self.problem_size = Some((vars, cons));
                true
            }

            Msg::StartSimplexTableau(tableau, maximization, method) => {
                self.solver_data = SolverData::SimplexTableau {
                    tableau: tableau.clone(),
                    method,
                };
                self.maximization = maximization;
                self.iterations.push(tableau); // Save initial tableau for iterations
                self.step = 0;
                true
            }

            Msg::StartSimplexMatrix(a, b, c) => {
                let m = b.len(); // number of constraints
                let n = c.len(); // number of variables (incl. any slack if you included them in 'a')

                // Suppose we assume the last `m` columns are slack => basis indices are [n-m, n-m+1, ... n-1].
                // If your input doesn't add slack columns, adapt as needed.
                let mut basis_indices = Vec::new();
                let start_col = n.saturating_sub(m);
                for col in start_col..n {
                    basis_indices.push(col);
                }

                self.solver_data = SolverData::SimplexMatrix {
                    a_matrix: a.clone(),
                    b_vector: b.clone(),
                    c_vector: c.clone(),
                    x_vector: DVector::zeros(n),
                    b_inverse: DMatrix::identity(m, m),
                    c_b_vector: DVector::zeros(m),
                    basis_indices,
                };
                self.iterations.clear();
                self.step = 0;
                true
            }

            Msg::StartInteriorPoint(a, b, c, alpha, initial) => {
                // Typically the user’s initial point has length == number of variables:
                let n = a.ncols();

                // If the user’s “initial” vector is empty or the wrong length, you can fallback:
                let feasible_x = if initial.len() == n {
                    DVector::from_vec(initial.clone())
                } else {
                    // Fallback: fill with 1.0 if the user input is missing
                    DVector::from_element(n, 1.0)
                };

                self.solver_data = SolverData::InteriorPoint {
                    a_matrix: a.clone(),
                    b_vector: b.clone(),
                    c_vector: c.clone(),

                    d_matrix: DMatrix::zeros(a.nrows(), a.ncols()),
                    a_tilde_matrix: DMatrix::zeros(a.nrows(), a.ncols()),
                    c_tilde_vector: DVector::zeros(a.ncols()),
                    p_matrix: DMatrix::identity(a.ncols(), a.ncols()),
                    cp_vector: DVector::zeros(b.len()),

                    // Now use the user’s feasible_x
                    current_x: feasible_x,

                    initial_point: initial.clone(), // Keep if you want to store it
                    alpha,
                };
                self.interior_iterations.clear();
                self.step = 0;
                true
            }

            Msg::NextStep => {
                match &mut self.solver_data {
                    SolverData::SimplexTableau { tableau, method } => {
                        if let Some(pivot_col) = find_pivot_column(tableau) {
                            if let Some(pivot_row) = find_pivot_row(tableau, pivot_col) {
                                perform_row_operations(tableau, pivot_row, pivot_col);
                                self.iterations.push(tableau.clone());
                                self.step += 1;
                            } else {
                                log::warn!("No valid pivot row found.");
                            }
                        } else {
                            log::warn!("No valid pivot column found.");
                        }
                    }
                    SolverData::SimplexMatrix {
                        a_matrix,
                        b_vector,
                        c_vector,
                        x_vector,
                        b_inverse,
                        c_b_vector,
                        basis_indices,
                    } => {
                        let mut problem = SimplexMatrixProblem {
                            a_matrix: a_matrix.clone(),
                            b_vector: b_vector.clone(),
                            c_vector: c_vector.clone(),
                            x_vector: x_vector.clone(),
                            b_inverse: b_inverse.clone(),
                            c_b_vector: c_b_vector.clone(),
                            basis_indices: basis_indices.clone(),
                        };

                        match perform_simplex_matrix_iteration(&mut problem) {
                            Ok(pivoted) => {
                                if pivoted {
                                    // Write updated data back
                                    *a_matrix = problem.a_matrix.clone();
                                    *b_vector = problem.b_vector.clone();
                                    *c_vector = problem.c_vector.clone();
                                    *x_vector = problem.x_vector.clone();
                                    *b_inverse = problem.b_inverse.clone();
                                    *c_b_vector = problem.c_b_vector.clone();
                                    *basis_indices = problem.basis_indices.clone();

                                    if let Ok(iter_data) = calculate_matrix_iteration(&problem) {
                                        self.matrix_iterations.push(iter_data);
                                    }

                                    self.step += 1;
                                    // Optionally push something into self.iterations if you want
                                    // to record or display each iteration snapshot.
                                } else {
                                    log::info!("MatrixForm: It's already optimal!");
                                }
                            }
                            Err(MatrixSimplexError::Unbounded(msg)) => {
                                log::error!("MatrixForm: Unbounded. {}", msg);
                                // You might set an error state or do something else
                            }
                            Err(e) => {
                                log::error!("MatrixForm iteration error: {:?}", e);
                            }
                        }
                    }
                    SolverData::InteriorPoint {
                        a_matrix,
                        b_vector,
                        c_vector,
                        current_x,
                        alpha,
                        d_matrix,
                        a_tilde_matrix,
                        c_tilde_vector,
                        p_matrix,
                        cp_vector,
                        initial_point,
                    } => {
                        // Build an InteriorPointProblem with `current_x` as the starting point
                        let mut problem = InteriorPointProblem {
                            a_matrix: a_matrix.clone(),
                            b_vector: b_vector.clone(),
                            c_vector: c_vector.clone(),
                            x_vector: current_x.clone(), // <--- user’s “feasible_x” from above
                            alpha: *alpha,
                            constraint_types: vec![],
                        };

                        // Perform one iteration
                        match perform_interior_point_iteration(&mut problem) {
                            Ok(iteration_data) => {
                                // iteration_data is an InteriorPointIteration
                                // 1) write it back into solver_data
                                *d_matrix = iteration_data.d_matrix.clone();
                                *a_tilde_matrix = iteration_data.a_tilde_matrix.clone();
                                *c_tilde_vector = iteration_data.c_tilde_vector.clone();
                                *p_matrix = iteration_data.p_matrix.clone();
                                *cp_vector = iteration_data.cp_vector.clone();
                                *current_x = iteration_data.current_x.clone(); // new x

                                // 2) Push the iteration snapshot so we can display it
                                self.interior_iterations.push(iteration_data);

                                self.step += 1;
                            }
                            Err(InteriorPointError::NoImprovement) => {
                                log::info!("No further improvement possible - likely optimal.");
                            }
                            Err(e) => {
                                log::error!("Interior Point iteration failed: {:?}", e);
                            }
                        }
                    }
                    SolverData::None => {
                        log::warn!("NextStep called with no solver selected.");
                    }

                    _ => {
                        log::warn!("NextStep called with unsupported solver type.");
                    }
                }
                true
            }

            Msg::Reset => {
                self.solver_data = SolverData::None;
                self.iterations.clear();
                self.matrix_iterations.clear();
                self.interior_iterations.clear(); // new
                self.step = 0;
                true
            }

            Msg::BackToSelection => {
                self.solver_type = SolverType::None;
                self.solver_data = SolverData::None;
                self.iterations.clear();
                self.step = 0;
                true
            }

            Msg::SetInitialPoint(x) => {
                // Use this x to set your `current_x` in SolverData::InteriorPoint, for instance
                if let SolverData::InteriorPoint { current_x, .. } = &mut self.solver_data {
                    *current_x = x;
                } else {
                    log::warn!("SetInitialPoint called, but solver is not InteriorPoint.");
                }
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let link = ctx.link();

        html! {
            <div class="app-container">
                <h1>{ "Linear Programming Solver" }</h1>

                if self.solver_type == SolverType::None {
                    // Solver selection interface
                    <div class="solver-selection">
                        <button
                            class="solver-button"
                            onclick={link.callback(|_| Msg::SelectSolver(SolverType::SimplexTableau))}>
                            { "Simplex Tableau Solver" }
                        </button>
                        <button
                            class="solver-button"
                            onclick={link.callback(|_| Msg::SelectSolver(SolverType::SimplexMatrix))}>
                            { "Simplex Matrix Solver" }
                        </button>
                        <button
                            class="solver-button"
                            onclick={link.callback(|_| Msg::SelectSolver(SolverType::InteriorPoint))}>
                            { "Interior Point Solver" }
                        </button>
                        <button
                            class="solver-button"
                            onclick={link.callback(|_| Msg::SelectSolver(SolverType::SimplexGrapher))}>
                            { "Simplex Grapher" }
                        </button>
                    </div>
                } else {
                    // Solver-specific views
                    <div>
                        <button
                            class="back-button"
                            onclick={link.callback(|_| Msg::BackToSelection)}>
                            { "← Back to Solver Selection" }
                        </button>

                        <InputForm
                            on_submit={link.callback(|input: InputFormData| match input {
                                InputFormData::TableauInput(matrix, is_max, method) =>
                                    Msg::StartSimplexTableau(matrix, is_max, method),
                                InputFormData::MatrixInput(a, b, c) =>
                                    Msg::StartSimplexMatrix(a, b, c),
                                InputFormData::InteriorPointInput(a, b, c, alpha, initial) =>
                                    Msg::StartInteriorPoint(a, b, c, alpha, initial),
                            })}
                            on_size_change={link.callback(|(vars, cons)| Msg::SetProblemSize(vars, cons))}
                            solver_type={self.solver_type.clone()}
                        />

                        { match &self.solver_data {
                            SolverData::SimplexTableau { tableau, method } => html! {
                                <div class="solver-results">
                                    <div class="controls">
                                        <button onclick={link.callback(|_| Msg::NextStep)}>{ "Next Step" }</button>
                                        <button onclick={link.callback(|_| Msg::Reset)}>{ "Reset" }</button>
                                    </div>
                                    <div class="iterations">
                                        { for self.iterations.iter().enumerate().map(|(i, tableau)| {
                                            html! {
                                                <div class="iteration">
                                                    <h3>{ format!("Iteration {}", i) }</h3>
                                                    <TableauView
                                                        tableau={tableau.clone()}
                                                        method={method.clone()}
                                                    />
                                                </div>
                                            }
                                        }) }
                                    </div>
                                </div>
                            },
                            SolverData::SimplexMatrix { .. } => html! {
                                <div class="solver-results">
                                    <div class="controls">
                                        <button onclick={link.callback(|_| Msg::NextStep)}>{ "Next Step" }</button>
                                        <button onclick={link.callback(|_| Msg::Reset)}>{ "Reset" }</button>
                                    </div>
                                    <div class="iterations">
                                        { for self.matrix_iterations.iter().enumerate().map(|(i, iter_data)| {
                                            html! {
                                                <SimplexMatrixView
                                                    iteration_number={i}
                                                    iteration_data={iter_data.clone()}
                                                />
                                            }
                                        }) }
                                    </div>
                                </div>
                            },
                            SolverData::InteriorPoint { .. } => html! {
                                <div class="solver-results">
                                    <div class="controls">
                                        <button onclick={link.callback(|_| Msg::NextStep)}>{ "Next Step" }</button>
                                        <button onclick={link.callback(|_| Msg::Reset)}>{ "Reset" }</button>
                                    </div>
                                    <div class="iterations">
                                        {
                                            for self.interior_iterations.iter().enumerate().map(|(i, iter_data)| {
                                                if self.interior_iterations.is_empty() {
                                                    // Show iteration=0 with no iteration_data
                                                    html! {
                                                        <InteriorPointView
                                                            iteration={0}
                                                            iteration_data={None}
                                                            problem_size={self.problem_size}
                                                            on_initial_point_submit={link.callback(|x: DVector<f64>| Msg::SetInitialPoint(x))}
                                                        />
                                                    }
                                                } else {
                                                    // We actually have iteration data, so display each snapshot
                                                    html! {
                                                        <div class="iterations">
                                                        {
                                                            for self.interior_iterations.iter().enumerate().map(|(i, iter_data)| {
                                                                html! {
                                                                    <InteriorPointView
                                                                        iteration={i}
                                                                        iteration_data={Some(iter_data.clone())}
                                                                    />
                                                                }
                                                            })
                                                        }
                                                        </div>
                                                    }
                                                }
                                            })
                                        }
                                    </div>
                                </div>
                            },
                            SolverData::None => html! {
                                <p>{ "No solver data available. Please start a solver." }</p>
                            },
                            _ => html! {
                                <p>{ "Solver not yet implemented." }</p>
                            }
                        }}
                    </div>
                }
            </div>
        }
    }
}
