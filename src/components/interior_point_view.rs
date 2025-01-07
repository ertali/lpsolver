use crate::interior_point::InteriorPointIteration;
use nalgebra::{DMatrix, DVector};
use web_sys::HtmlInputElement;
use yew::prelude::*;

/// The component’s props, analogous to your other solver views.
#[derive(Properties, PartialEq)]
pub struct Props {
    /// Which iteration index are we displaying (0, 1, 2, ...).
    pub iteration: usize,

    /// The current iteration data from the solver. If None, we display placeholders.
    pub iteration_data: Option<InteriorPointIteration>,

    /// An optional callback if you want the user to input an initial feasible point
    /// at iteration == 0. If you don’t need user input, remove this.
    #[prop_or_default]
    pub on_initial_point_submit: Option<Callback<DVector<f64>>>,

    /// (vars, cons) so we know how many variables if we let the user input an initial point.
    /// If None, default to 1 variable.
    #[prop_or(None)]
    pub problem_size: Option<(usize, usize)>,
}

/// Messages for user input (initial feasible point).
pub enum Msg {
    UpdateInitialPoint(usize, f64),
    SubmitInitialPoint,
}

/// The `InteriorPointView` keeps local state for the initial feasible point (if iteration=0).
pub struct InteriorPointView {
    initial_point: Vec<f64>,
}

impl Component for InteriorPointView {
    type Message = Msg;
    type Properties = Props;

    fn create(ctx: &Context<Self>) -> Self {
        // Default the initial point to 1.0 in each variable
        let var_count = ctx.props().problem_size.map(|(vars, _)| vars).unwrap_or(1);
        Self {
            initial_point: vec![1.0; var_count],
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::UpdateInitialPoint(idx, val) => {
                if idx < self.initial_point.len() {
                    self.initial_point[idx] = val;
                    true // re-render so user sees updated fields
                } else {
                    false
                }
            }
            Msg::SubmitInitialPoint => {
                let x = DVector::from_vec(self.initial_point.clone());
                // If the parent provided on_initial_point_submit, call it
                if let Some(cb) = &ctx.props().on_initial_point_submit {
                    cb.emit(x);
                }
                // No need to re-render
                false
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let props = ctx.props();

        // Decompose iteration_data for easy references
        let (d_matrix, a_tilde, c_tilde, p_matrix, cp_vector, current_x) =
            if let Some(iter_data) = &props.iteration_data {
                (
                    Some(&iter_data.d_matrix),
                    Some(&iter_data.a_tilde_matrix),
                    Some(&iter_data.c_tilde_vector),
                    Some(&iter_data.p_matrix),
                    Some(&iter_data.cp_vector),
                    Some(&iter_data.current_x),
                )
            } else {
                (None, None, None, None, None, None)
            };

        html! {
            <div class="interior-point-view">
                <h3>{ format!("Iteration {}", props.iteration) }</h3>

                // If we're at iteration 0, let user specify an initial feasible x
                if props.iteration == 0 {
                    <div class="initial-point-input">
                        <h4>{"Initial Feasible Point"}</h4>
                        <div class="initial-point-values">
                            {
                                for (0..self.initial_point.len()).map(|i| {
                                    html! {
                                        <label>
                                            {format!("x{} = ", i + 1)}
                                            <input
                                                type="number"
                                                step="0.1"
                                                value={self.initial_point[i].to_string()}
                                                oninput={ctx.link().callback(move |e: InputEvent| {
                                                    let input: HtmlInputElement = e.target_unchecked_into();
                                                    Msg::UpdateInitialPoint(i, input.value().parse().unwrap_or(1.0))
                                                })}
                                            />
                                        </label>
                                    }
                                })
                            }
                        </div>
                        <button onclick={ctx.link().callback(|_| Msg::SubmitInitialPoint)}>
                            {"Start Interior Point Method"}
                        </button>
                        <p class="initial-point-description">
                            {"Enter initial values for all variables (ideally satisfying constraints)."}
                        </p>
                    </div>
                }

                // Now display the iteration's D, A~, c~, P, Pc~, and x
                <div class="matrix-container">
                    <div class="matrix-box">
                        <h4>{"D Matrix (diag(x))"}</h4>
                        { self.render_matrix(d_matrix) }
                    </div>

                    <div class="matrix-box">
                        <h4>{"A~ = A * D"}</h4>
                        { self.render_matrix(a_tilde) }
                    </div>

                    <div class="matrix-box">
                        <h4>{"c~ = D * c"}</h4>
                        { self.render_vector(c_tilde) }
                    </div>

                    <div class="matrix-box">
                        <h4>{"P = I - A~^T (A~ A~^T)^{-1} A~"}</h4>
                        { self.render_matrix(p_matrix) }
                    </div>

                    <div class="matrix-box">
                        <h4>{"P c~"}</h4>
                        { self.render_vector(cp_vector) }
                    </div>

                    <div class="matrix-box">
                        <h4>{"Current x"}</h4>
                        { self.render_vector(current_x) }
                    </div>
                </div>
            </div>
        }
    }
}

impl InteriorPointView {
    /// Renders an (m x n) matrix in a table, or a placeholder if None
    fn render_matrix(&self, matrix_opt: Option<&DMatrix<f64>>) -> Html {
        if let Some(mat) = matrix_opt {
            let (rows, cols) = mat.shape();
            html! {
                <table class="matrix">
                    <tbody>
                    {
                        for (0..rows).map(|r| html! {
                            <tr>
                            {
                                for (0..cols).map(|c| html! {
                                    <td>{ format!("{:.4}", mat[(r,c)]) }</td>
                                })
                            }
                            </tr>
                        })
                    }
                    </tbody>
                </table>
            }
        } else {
            html! { <p>{"Matrix not available"}</p> }
        }
    }

    /// Renders an (n x 1) vector in a vertical table, or a placeholder if None
    fn render_vector(&self, vec_opt: Option<&DVector<f64>>) -> Html {
        if let Some(v) = vec_opt {
            html! {
                <table class="vector">
                    <tbody>
                    {
                        for (0..v.len()).map(|i| html! {
                            <tr>
                                <td>{ format!("{:.4}", v[i]) }</td>
                            </tr>
                        })
                    }
                    </tbody>
                </table>
            }
        } else {
            html! { <p>{"Vector not available"}</p> }
        }
    }
}
