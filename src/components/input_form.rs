use crate::components::SolverType;
use crate::simplex::*;
use nalgebra::{DMatrix, DVector};
use wasm_bindgen::JsCast;
use web_sys::{HtmlInputElement, HtmlSelectElement, HtmlTextAreaElement};
use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct Props {
    pub on_submit: Callback<(InputFormData)>,
    pub on_size_change: Callback<(usize, usize)>,
    #[prop_or(SolverType::SimplexTableau)]
    pub solver_type: SolverType,
    #[prop_or(10)]
    pub max_variables: usize,
}

#[derive(Clone)]
pub enum InputFormData {
    TableauInput(DMatrix<f64>, bool, String),
    MatrixInput(DMatrix<f64>, DVector<f64>, DVector<f64>),
    InteriorPointInput(DMatrix<f64>, DVector<f64>, DVector<f64>, f64, Vec<f64>),
}

#[derive(Clone, PartialEq)]
pub enum SolverConfig {
    SimplexTableau {
        method: String,
    },
    SimplexMatrix,
    InteriorPoint {
        alpha: f64,
        initial_feasible: Vec<f64>,
    },
}

pub struct InputForm {
    variables: usize,
    constraints: usize,
    objective_coeffs: Vec<f64>,
    constraint_coeffs: Vec<Vec<f64>>,
    constraint_signs: Vec<String>,
    rhs_values: Vec<f64>,
    maximization: bool,
    config: SolverConfig,
}

pub enum Msg {
    SetVariables(usize),
    SetConstraints(usize),
    UpdateObjectiveCoeff(usize, f64),
    UpdateConstraintCoeff(usize, usize, f64),
    UpdateConstraintSign(usize, String),
    UpdateRHSValue(usize, f64),
    UpdateMethod(String),
    UpdateAlpha(f64),
    ToggleOptimizationType,
    Submit,
    UpdateInitialPoint(usize, f64),
}

impl Component for InputForm {
    type Message = Msg;
    type Properties = Props;

    fn create(ctx: &Context<Self>) -> Self {
        let config = match ctx.props().solver_type {
            SolverType::SimplexTableau => SolverConfig::SimplexTableau {
                method: "big_m".to_string(),
            },
            SolverType::SimplexMatrix => SolverConfig::SimplexMatrix,
            SolverType::InteriorPoint => SolverConfig::InteriorPoint {
                alpha: 0.5,
                initial_feasible: vec![],
            },
            _ => SolverConfig::SimplexTableau {
                method: "big_m".to_string(),
            },
        };

        Self {
            variables: 2,
            constraints: 2,
            objective_coeffs: vec![0.0; 2],
            constraint_coeffs: vec![vec![0.0; 2]; 2],
            constraint_signs: vec!["<=".to_string(); 2],
            rhs_values: vec![0.0; 2],
            maximization: true,
            config,
        }
    }

    /*
    fn changed(&mut self, ctx: &Context<Self>, _old_props: &Self::Properties) -> bool {
        let new_config = match ctx.props().solver_type {
            SolverType::SimplexTableau => SolverConfig::SimplexTableau {
                method: "big_m".to_string(),
            },
            SolverType::SimplexMatrix => SolverConfig::SimplexMatrix,
            SolverType::InteriorPoint => SolverConfig::InteriorPoint {
                alpha: 0.5,
                initial_feasible: vec![],
            },
            _ => SolverConfig::SimplexTableau {
                method: "big_m".to_string(),
            },
        };

        if self.config != new_config {
            self.config = new_config;
            true
        } else {
            false
        }
    }*/

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::SetVariables(v) => {
                let max_vars = ctx.props().max_variables;
                self.variables = v.min(max_vars);
                self.update_sizes();
                ctx.props()
                    .on_size_change
                    .emit((self.variables, self.constraints));
                true
            }
            Msg::SetConstraints(c) => {
                self.constraints = c;
                self.update_sizes();
                ctx.props()
                    .on_size_change
                    .emit((self.variables, self.constraints));
                true
            }
            Msg::UpdateObjectiveCoeff(idx, val) => {
                if idx < self.objective_coeffs.len() {
                    self.objective_coeffs[idx] = val;
                    true
                } else {
                    false
                }
            }
            Msg::UpdateConstraintCoeff(i, j, val) => {
                if i < self.constraint_coeffs.len() && j < self.constraint_coeffs[i].len() {
                    self.constraint_coeffs[i][j] = val;
                    true
                } else {
                    false
                }
            }
            Msg::UpdateConstraintSign(idx, sign) => {
                if idx < self.constraint_signs.len() {
                    self.constraint_signs[idx] = sign;
                    let needs_artificial = self.has_artificial_variables();
                    if let SolverConfig::SimplexTableau { method } = &mut self.config {
                        if needs_artificial {
                            if method.is_empty() {
                                *method = "big_m".to_string();
                            }
                        }
                    }
                    true
                } else {
                    false
                }
            }
            Msg::UpdateRHSValue(idx, val) => {
                if idx < self.rhs_values.len() {
                    self.rhs_values[idx] = val;
                    true
                } else {
                    false
                }
            }
            Msg::Submit => {
                self.submit(ctx);
                true
            }
            Msg::UpdateMethod(new_method) => {
                let needs_artificial = self.has_artificial_variables();
                if let SolverConfig::SimplexTableau { method } = &mut self.config {
                    *method = new_method;
                    if needs_artificial {
                        true
                    } else {
                        *method = "Big-M".to_string();
                        true
                    }
                } else {
                    false
                }
            }
            Msg::ToggleOptimizationType => {
                self.maximization = !self.maximization;
                true
            }
            Msg::UpdateAlpha(new_alpha) => {
                if let SolverConfig::InteriorPoint { ref mut alpha, .. } = self.config {
                    *alpha = new_alpha.max(0.0).min(1.0);
                    true
                } else {
                    false
                }
            }
            Msg::UpdateInitialPoint(idx, val) => {
                if let SolverConfig::InteriorPoint {
                    ref mut initial_feasible,
                    ..
                } = self.config
                {
                    if idx >= initial_feasible.len() {
                        initial_feasible.resize(idx + 1, 0.0);
                    }
                    initial_feasible[idx] = val;
                    true
                } else {
                    false
                }
            }
        }
    }
    fn view(&self, ctx: &Context<Self>) -> Html {
        let link = ctx.link();

        html! {
            <div class="input-form">
                <div class="optimization-type">
                    <select
                        value={if self.maximization { "max" } else { "min" }}
                        onchange={link.callback(|e: Event| {
                            let select: HtmlSelectElement = e.target_unchecked_into();
                            Msg::ToggleOptimizationType
                        })}>
                        <option value="min">{"Minimize"}</option>
                        <option value="max">{"Maximize"}</option>
                    </select>
                    <span>{" Z = "}</span>
                </div>

                <div class="size-selectors">
                    <div>
                        <label>{"Variables: "}
                            <input
                                type="number"
                                min="1"
                                max={ctx.props().max_variables.to_string()}
                                value={self.variables.to_string()}
                                onchange={link.callback(|e: Event| {
                                    let input: HtmlInputElement = e.target_unchecked_into();
                                    Msg::SetVariables(input.value().parse().unwrap_or(2))
                                })}
                            />
                        </label>
                    </div>
                    <div>
                        <label>{"Constraints: "}
                            <input
                                type="number"
                                min="1"
                                max="10"
                                value={self.constraints.to_string()}
                                onchange={link.callback(|e: Event| {
                                    let input: HtmlInputElement = e.target_unchecked_into();
                                    Msg::SetConstraints(input.value().parse().unwrap_or(2))
                                })}
                            />
                        </label>
                    </div>
                </div>

                <div class="matrix-input">
                    <div class="objective-function">
                        {for (0..self.variables).map(|j| {
                            html! {
                                <span>
                                    {if j > 0 { " + " } else { "" }}
                                    <input
                                        type="number"
                                        step="0.1"
                                        value={self.objective_coeffs[j].to_string()}
                                        onchange={link.callback(move |e: Event| {
                                            let input: HtmlInputElement = e.target_unchecked_into();
                                            Msg::UpdateObjectiveCoeff(j, input.value().parse().unwrap_or(0.0))
                                        })}
                                    />
                                    {format!("x{}", j + 1)}
                                </span>
                            }
                        })}
                    </div>

                    <div class="constraints">
                        {for (0..self.constraints).map(|i| {
                            html! {
                                <div class="constraint-row">
                                    {for (0..self.variables).map(|j| {
                                        html! {
                                            <span>
                                                {if j > 0 { " + " } else { "" }}
                                                <input
                                                    type="number"
                                                    step="0.1"
                                                    value={self.constraint_coeffs[i][j].to_string()}
                                                    onchange={link.callback(move |e: Event| {
                                                        let input: HtmlInputElement = e.target_unchecked_into();
                                                        Msg::UpdateConstraintCoeff(i, j, input.value().parse().unwrap_or(0.0))
                                                    })}
                                                />
                                                {format!("x{}", j + 1)}
                                            </span>
                                        }
                                    })}
                                    <select
                                        value={self.constraint_signs[i].clone()}
                                        onchange={link.callback(move |e: Event| {
                                            let select: HtmlSelectElement = e.target_unchecked_into();
                                            Msg::UpdateConstraintSign(i, select.value())
                                        })}>
                                        <option value=">=">{"≥"}</option>
                                        <option value="=">{"="}</option>
                                        <option value="<=">{"≤"}</option>
                                    </select>
                                    <input
                                        type="number"
                                        step="0.1"
                                        value={self.rhs_values[i].to_string()}
                                        onchange={link.callback(move |e: Event| {
                                            let input: HtmlInputElement = e.target_unchecked_into();
                                            Msg::UpdateRHSValue(i, input.value().parse().unwrap_or(0.0))
                                        })}
                                    />
                                </div>
                            }
                        })}
                    </div>
                </div>

                {match &self.config {
                    SolverConfig::SimplexTableau { method } => {
                        let needs_artificial = self.has_artificial_variables();
                        html! {
                            if needs_artificial {
                                <div class="method-selector">
                                    <label>{"Solution Method:"}</label>
                                    <select
                                        value={method.clone()}
                                        onchange={link.callback(|e: Event| {
                                            let select: HtmlSelectElement = e.target_unchecked_into();
                                            Msg::UpdateMethod(select.value())
                                        })}>
                                        <option value="big_m">{"Big-M Method"}</option>
                                        <option value="two_phase">{"Two-Phase Method"}</option>
                                    </select>
                                    <div class="method-description">
                                        {"Choose between Big-M or Two-Phase method for handling artificial variables."}
                                    </div>
                                </div>
                            }
                        }
                    },
                    SolverConfig::InteriorPoint { alpha, initial_feasible } => html! {
                        <>
                            <div class="initial-point-input">
                                <h4>{"Initial Feasible Point"}</h4>
                                <div class="initial-point-values">
                                    {for (0..self.variables).map(|i| {
                                        html! {
                                            <label>
                                                {format!("x{} = ", i + 1)}
                                                <input
                                                    type="number"
                                                    step="0.1"
                                                    value={initial_feasible.get(i).cloned().unwrap_or(0.0).to_string()}
                                                    onchange={link.callback(move |e: Event| {
                                                        let input: HtmlInputElement = e.target_unchecked_into();
                                                        Msg::UpdateInitialPoint(i, input.value().parse().unwrap_or(0.0))
                                                    })}
                                                />
                                            </label>
                                        }
                                    })}
                                </div>
                                <div class="initial-point-description">
                                    {"Enter an initial feasible point that satisfies all constraints."}
                                </div>
                            </div>

                            <div class="alpha-selector">
                                <label>{"Step Size (α): "}
                                    <input
                                        type="number"
                                        min="0"
                                        max="1"
                                        step="0.1"
                                        value={alpha.to_string()}
                                        onchange={
                                            let alpha = *alpha;
                                            link.callback(move |e: Event| {
                                            let input: HtmlInputElement = e.target_unchecked_into();
                                            Msg::UpdateAlpha(input.value().parse().unwrap_or(alpha))
                                        })}
                                    />
                                </label>
                                <div class="alpha-description">
                                    {"α controls the step size in each iteration (between 0 and 1)."}
                                </div>
                            </div>
                        </>
                    },
                    _ => html! {},
                }}
                <button onclick={ctx.link().callback(|_| Msg::Submit)}>
                    {"Solve"}
                </button>
            </div>
        }
    }
}

impl InputForm {
    fn update_sizes(&mut self) {
        self.objective_coeffs.resize(self.variables, 0.0);
        self.constraint_coeffs
            .resize(self.constraints, vec![0.0; self.variables]);
        for row in self.constraint_coeffs.iter_mut() {
            row.resize(self.variables, 0.0);
        }
        self.constraint_signs
            .resize(self.constraints, "<=".to_string());
        self.rhs_values.resize(self.constraints, 0.0);
    }

    fn has_artificial_variables(&self) -> bool {
        self.constraint_signs
            .iter()
            .any(|sign| sign == "=" || sign == ">=")
    }

    fn submit(&self, ctx: &Context<Self>) {
        match &self.config {
            SolverConfig::SimplexTableau { method } => {
                if let Some(tableau) = self.create_tableau() {
                    ctx.props().on_submit.emit(InputFormData::TableauInput(
                        tableau,
                        self.maximization,
                        method.clone(),
                    ));
                }
            }
            SolverConfig::SimplexMatrix => {
                if let Some((a, b, c)) = self.create_matrix_form() {
                    ctx.props()
                        .on_submit
                        .emit(InputFormData::MatrixInput(a, b, c));
                }
            }
            SolverConfig::InteriorPoint {
                alpha,
                initial_feasible,
            } => {
                if let Some((a, b, c)) = self.create_matrix_form() {
                    ctx.props()
                        .on_submit
                        .emit(InputFormData::InteriorPointInput(
                            a,
                            b,
                            c,
                            *alpha,
                            initial_feasible.clone(),
                        ));
                }
            }
        }
    }

    fn create_tableau(&self) -> Option<DMatrix<f64>> {
        if let SolverConfig::SimplexTableau { method } = &self.config {
            let artificial_handler = match method.as_str() {
                "big_m" => 0,
                "two_phase" => 1,
                _ => return None,
            };

            let problem = augment_tableau(
                self.objective_coeffs.clone(),
                self.constraint_coeffs.clone(),
                self.constraint_signs.clone(),
                self.rhs_values.clone(),
                artificial_handler,
            );

            Some(problem.tableau)
        } else {
            None
        }
    }

    fn create_matrix_form(&self) -> Option<(DMatrix<f64>, DVector<f64>, DVector<f64>)> {
        let m = self.constraints;
        let n = self.variables;

        let mut a_data = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                a_data.push(self.constraint_coeffs[i][j]);
            }
        }
        let a = DMatrix::from_row_slice(m, n, &a_data);

        let b = DVector::from_iterator(m, self.rhs_values.iter().cloned());

        let sign = if self.maximization { 1.0 } else { -1.0 };
        let c_vals: Vec<f64> = self
            .objective_coeffs
            .iter()
            .map(|&val| sign * val)
            .collect();
        let c = DVector::from_vec(c_vals);

        Some((a, b, c))
    }
}
