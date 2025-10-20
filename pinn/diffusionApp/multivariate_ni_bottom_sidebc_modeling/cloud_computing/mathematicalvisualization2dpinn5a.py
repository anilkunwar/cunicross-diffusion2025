import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ============================================================
# Utility Functions
# ============================================================

def compute_fluxes_and_grads(c1_preds, c2_preds, X, Y, params):
    """Compute fluxes and gradients."""
    dx = X[1] - X[0]
    dy = Y[1] - Y[0]
    grad_c1_y = np.gradient(c1_preds, dy, axis=1)
    grad_c2_y = np.gradient(c2_preds, dy, axis=1)
    D11 = params.get('D11', 1e-14)
    D22 = params.get('D22', 1e-14)
    J1_preds = -D11 * grad_c1_y
    J2_preds = -D22 * grad_c2_y
    return J1_preds, J2_preds, grad_c1_y, grad_c2_y


def get_plot_customization():
    """Sidebar configuration options for consistent styling."""
    st.sidebar.header("Plot Customization")

    color_cu = st.sidebar.color_picker("Cu curve color", "#1f77b4")
    color_ni = st.sidebar.color_picker("Ni curve color", "#d62728")
    line_width = st.sidebar.slider("Line width", 1.0, 6.0, 2.8)
    line_style = st.sidebar.selectbox("Line style", ["solid", "dot", "dash", "longdash"])
    fig_width = st.sidebar.slider("Figure width (inches)", 6, 14, 8)
    fig_height = st.sidebar.slider("Figure height (inches)", 4, 10, 6)
    font_size = st.sidebar.slider("Font size", 10, 30, 16)
    colorscale_plotly = st.sidebar.selectbox("Plotly colormap", ["Viridis", "Inferno", "Plasma", "Cividis", "Turbo"])

    return (color_cu, color_ni, line_width, line_style, fig_width, fig_height, font_size, colorscale_plotly)

# ============================================================
# Visualization Functions
# ============================================================

def plot_flux_vs_gradient_plotly(solution, time_index, color_cu, color_ni,
                                 line_width, line_style, fig_width, fig_height, font_size):
    """Plot Flux vs Gradient curve with shaded uphill diffusion regions."""
    J1 = solution['J1_preds'][time_index].flatten()
    J2 = solution['J2_preds'][time_index].flatten()
    grad_c1 = solution['grad_c1_y'][time_index].flatten()
    grad_c2 = solution['grad_c2_y'][time_index].flatten()

    uphill_mask_cu = J1 * grad_c1 > 0
    uphill_mask_ni = J2 * grad_c2 > 0

    fig = go.Figure()

    # Cu line
    fig.add_trace(go.Scatter(
        x=grad_c1, y=J1, mode='lines', name='Cu: J₁ vs ∇c₁',
        line=dict(color=color_cu, width=line_width, dash=line_style)
    ))

    # Ni line
    fig.add_trace(go.Scatter(
        x=grad_c2, y=J2, mode='lines', name='Ni: J₂ vs ∇c₂',
        line=dict(color=color_ni, width=line_width, dash=line_style)
    ))

    # Uphill region shading
    fig.add_trace(go.Scatter(
        x=grad_c1[uphill_mask_cu], y=J1[uphill_mask_cu],
        mode='markers', name='Uphill (Cu)',
        marker=dict(color='rgba(31,119,180,0.5)', size=6, symbol='circle')
    ))

    fig.add_trace(go.Scatter(
        x=grad_c2[uphill_mask_ni], y=J2[uphill_mask_ni],
        mode='markers', name='Uphill (Ni)',
        marker=dict(color='rgba(214,39,40,0.5)', size=6, symbol='circle')
    ))

    fig.update_layout(
        title=f"Flux vs Gradient Curve at t={solution['times'][time_index]:.2f}s",
        xaxis_title="∇cᵢ (1/m)",
        yaxis_title="Jᵢ (mol/m²·s)",
        font=dict(size=font_size),
        width=fig_width * 100,
        height=fig_height * 100,
        template="plotly_white",
        legend=dict(title="Legend", font=dict(size=font_size-2))
    )
    st.plotly_chart(fig, use_container_width=True)

    # Caption
    st.markdown("""
    **Caption:**  
    The *Flux vs Gradient* curve illustrates how the diffusion flux \(J_i\) relates to its concentration gradient \(\nabla c_i\).  
    Uphill diffusion regions (shaded) indicate where the flux flows **against** the gradient (i.e., \(J_i \nabla c_i > 0\)),  
    revealing anomalous or coupled diffusion behavior.
    """)


def plot_uphill_time_evolution(solution, color_cu, color_ni, line_width, fig_width, fig_height, font_size):
    """Plot temporal evolution of the global maxima of J_i * ∇c_i."""
    times = np.array(solution['times'])
    J1dot = [np.max(J * g) for J, g in zip(solution['J1_preds'], solution['grad_c1_y'])]
    J2dot = [np.max(J * g) for J, g in zip(solution['J2_preds'], solution['grad_c2_y'])]

    df = pd.DataFrame({
        "Time (s)": times,
        "max(J₁·∇c₁)": J1dot,
        "max(J₂·∇c₂)": J2dot
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=J1dot, mode='lines+markers', name='Cu', line=dict(color=color_cu, width=line_width)))
    fig.add_trace(go.Scatter(x=times, y=J2dot, mode='lines+markers', name='Ni', line=dict(color=color_ni, width=line_width)))

    fig.update_layout(
        title="Temporal Evolution of max(Jᵢ·∇cᵢ)",
        xaxis_title="Time (s)",
        yaxis_title="max(Jᵢ·∇cᵢ)",
        font=dict(size=font_size),
        width=fig_width * 100,
        height=fig_height * 100,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Caption:**  
    This plot tracks the **maximum product \(J_i·∇c_i\)** over time.  
    Peaks represent moments where the **uphill diffusion tendency** is most pronounced,  
    indicating strong coupling effects or driving forces overcoming the concentration gradient.
    """)

    st.dataframe(df.style.highlight_max(axis=0, color='lightgreen'))


def plot_uphill_heatmap(solution, time_index, colorscale_plotly, fig_width, fig_height, font_size):
    """2D heatmap of uphill diffusion intensity."""
    J1 = solution['J1_preds'][time_index]
    grad_c1 = solution['grad_c1_y'][time_index]
    uphill_field = J1 * grad_c1

    fig = go.Figure(data=go.Heatmap(
        z=uphill_field,
        colorscale=colorscale_plotly,
        colorbar_title="J₁·∇c₁"
    ))

    fig.update_layout(
        title=f"Uphill Diffusion Heatmap (t={solution['times'][time_index]:.2f}s)",
        xaxis_title="X position (μm)",
        yaxis_title="Y position (μm)",
        font=dict(size=font_size),
        width=fig_width * 100,
        height=fig_height * 100,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Caption:**  
    The heatmap shows the spatial distribution of \(J₁·∇c₁\).  
    Positive regions correspond to **uphill diffusion zones**,  
    where flux opposes the local concentration gradient.
    """)

# ============================================================
# Main Streamlit App
# ============================================================

def main():
    st.title("📊 Multicomponent Diffusion Visualization Suite")

    # Synthetic example (replace with real data loading)
    Nx, Ny, Nt = 40, 40, 10
    X, Y = np.linspace(0, 10, Nx), np.linspace(0, 5, Ny)
    c1_preds = np.random.rand(Nt, Nx, Ny)
    c2_preds = np.random.rand(Nt, Nx, Ny)
    params = {'D11': 1e-14, 'D22': 2e-14, 'Ly': 5, 'Lx': 10}
    times = np.linspace(0, 100, Nt)

    solution = {'X': X, 'Y': Y, 'c1_preds': c1_preds, 'c2_preds': c2_preds,
                'params': params, 'times': times}

    # Compute fluxes and gradients
    J1, J2, grad_c1, grad_c2 = compute_fluxes_and_grads(c1_preds, c2_preds, X, Y, params)
    solution.update({'J1_preds': J1, 'J2_preds': J2, 'grad_c1_y': grad_c1, 'grad_c2_y': grad_c2})

    # Sidebar
    time_index = st.sidebar.slider("Time Index", 0, len(times)-1, 0)
    (color_cu, color_ni, line_width, line_style, fig_width, fig_height, font_size, colorscale_plotly) = get_plot_customization()

    st.header("1️⃣ Flux vs Gradient Curve")
    plot_flux_vs_gradient_plotly(solution, time_index, color_cu, color_ni, line_width, line_style, fig_width, fig_height, font_size)

    st.header("2️⃣ Temporal Evolution of max(Jᵢ·∇cᵢ)")
    plot_uphill_time_evolution(solution, color_cu, color_ni, line_width, fig_width, fig_height, font_size)

    st.header("3️⃣ Uphill Diffusion Heatmap")
    plot_uphill_heatmap(solution, time_index, colorscale_plotly, fig_width, fig_height, font_size)

    st.success("✅ Visualization Complete — publication-quality figures ready.")

# ============================================================
# Execute
# ============================================================

if __name__ == "__main__":
    main()
