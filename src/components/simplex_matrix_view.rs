use crate::simplex_matrix::SimplexMatrixIteration;
use nalgebra::{DMatrix, DVector};
use yew::prelude::*;

/// Props for displaying a single matrix-form Simplex iteration.
#[derive(Properties, PartialEq)]
pub struct Props {
    /// Which iteration number is this? (for display)
    pub iteration_number: usize,

    /// The iteration data (a snapshot of the solver state).
    pub iteration_data: SimplexMatrixIteration,
}

/// A component to display one iteration of the matrix-based Simplex solver.
pub struct SimplexMatrixView;

impl Component for SimplexMatrixView {
    type Message = ();
    type Properties = Props;

    fn create(_ctx: &Context<Self>) -> Self {
        Self
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let iteration_number = ctx.props().iteration_number;
        let data = &ctx.props().iteration_data;

        // Decompose the data for convenience:
        let a_matrix = &data.a_matrix;
        let b_vector = &data.b_vector;
        let c_vector = &data.c_vector;
        let z_value = data.z_value;
        let x_vector = &data.x_vector;
        let x_s_vector = &data.x_s_vector;
        let cp_b_inverse_a = &data.cp_b_inverse_a;
        let cp_b_inverse = &data.cp_b_inverse;
        let b_inverse_a = &data.b_inverse_a;
        let b_inverse = &data.b_inverse;

        html! {
            <div class="matrix-form-iteration">
                <h3>{ format!("Matrix Form - Iteration {}", iteration_number) }</h3>

                <h4>{ "A Matrix" }</h4>
                { self.render_matrix(a_matrix) }

                <h4>{ "b Vector (RHS)" }</h4>
                { self.render_vector(b_vector) }

                <h4>{ "Objective Coefficients (c)" }</h4>
                { self.render_vector(c_vector) }

                <div class="current-objective">
                    { format!("Current Objective Value (z) = {:.4}", z_value) }
                </div>

                <h4>{ "Current Solution (x)" }</h4>
                { self.render_solution(x_vector) }

                <h4>{ "Slack / Surplus Vector (x_s)" }</h4>
                { self.render_vector(x_s_vector) }

                <h4>{ "Reduced Costs (cB * B^-1 * A - c)" }</h4>
                { self.render_matrix(cp_b_inverse_a) }

                <h4>{ "cB * B^-1" }</h4>
                { self.render_matrix(cp_b_inverse) }

                <h4>{ "B^-1 * A" }</h4>
                { self.render_matrix(b_inverse_a) }

                <h4>{ "B^-1 (Basis Inverse)" }</h4>
                { self.render_matrix(b_inverse) }
            </div>
        }
    }
}

impl SimplexMatrixView {
    fn render_matrix(&self, matrix: &DMatrix<f64>) -> Html {
        let (rows, cols) = matrix.shape();
        html! {
            <table class="matrix">
                <tbody>
                {
                    for (0..rows).map(|i| html! {
                        <tr>
                            {
                                for (0..cols).map(|j| html! {
                                    <td>{ format!("{:.4}", matrix[(i,j)]) }</td>
                                })
                            }
                        </tr>
                    })
                }
                </tbody>
            </table>
        }
    }

    fn render_vector(&self, vector: &DVector<f64>) -> Html {
        html! {
            <table class="matrix">
                <tbody>
                {
                    for (0..vector.len()).map(|i| html! {
                        <tr>
                            <td>{ format!("{:.4}", vector[i]) }</td>
                        </tr>
                    })
                }
                </tbody>
            </table>
        }
    }

    fn render_solution(&self, x: &DVector<f64>) -> Html {
        html! {
            <div class="solution-vector">
            {
                for (0..x.len()).map(|i| html! {
                    <div>
                        { format!("x{} = {:.4}", i+1, x[i]) }
                    </div>
                })
            }
            </div>
        }
    }
}
