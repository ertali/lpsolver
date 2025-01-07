use nalgebra::DMatrix;
use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct Props {
    pub tableau: DMatrix<f64>,
    #[prop_or_default]
    pub highlight_pivot: Option<(usize, usize)>,
    #[prop_or_default]
    pub is_phase_one: bool,
    #[prop_or_default]
    pub method: String,
}

pub struct TableauView;

impl Component for TableauView {
    type Message = ();
    type Properties = Props;

    fn create(_ctx: &Context<Self>) -> Self {
        Self
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let tableau = &ctx.props().tableau;
        let highlight_pivot = &ctx.props().highlight_pivot;
        let is_phase_one = ctx.props().is_phase_one;
        let method = &ctx.props().method;
        let last_row = tableau.nrows() - 1;

        html! {
            <div class="tableau">
                <table>
                    <tbody>
                        {for (0..tableau.nrows()).map(|i| {
                            html! {
                                <tr class={if i == last_row { "objective-row" } else { "" }}>
                                    {for (0..tableau.ncols()).map(|j| {
                                        let is_pivot = highlight_pivot.map_or(false, |(pi, pj)| pi == i && pj == j);
                                        let value = tableau[(i, j)];
                                        let formatted_value = if method == "big_m" && !is_phase_one {
                                            format_big_m_value(value, i == last_row)
                                        } else {
                                            format_regular_value(value)
                                        };

                                        html! {
                                            <td class={if is_pivot { "pivot-cell" } else { "" }}>
                                                {formatted_value}
                                            </td>
                                        }
                                    })}
                                </tr>
                            }
                        })}
                    </tbody>
                </table>
            </div>
        }
    }
}

fn format_big_m_value(value: f64, is_objective_row: bool) -> String {
    let big_m = 10_000_000.0;

    // Check if this is a Big M coefficient
    if value.abs() > big_m - 1e-10 {
        let m_coeff = value / big_m;
        if m_coeff.abs() < 1e-10 {
            "0.00".to_string()
        } else if (m_coeff - 1.0).abs() < 1e-10 {
            if is_objective_row { "M" } else { "1.00" }.to_string()
        } else if (m_coeff + 1.0).abs() < 1e-10 {
            if is_objective_row { "-M" } else { "-1.00" }.to_string()
        } else {
            if is_objective_row {
                format!("{:.2}M", m_coeff)
            } else {
                format!("{:.2}", m_coeff)
            }
        }
    } else {
        format_regular_value(value)
    }
}

fn format_regular_value(value: f64) -> String {
    if value.abs() < 1e-10 {
        "0.00".to_string()
    } else {
        format!("{:.2}", value)
    }
}
