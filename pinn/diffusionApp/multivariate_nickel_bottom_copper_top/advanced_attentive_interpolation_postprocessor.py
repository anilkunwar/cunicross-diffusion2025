import os
import pickle
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

# ===================================
# Configuration
# ===================================
SOLUTION_DIR = Path(__file__).parent / "pinn_solutions"

# ===================================
# Utilities
# ===================================
@st.cache_data
def load_solutions(solution_dir):
    """Load all valid PINN solution files."""
    solutions = []
    lys = []
    for fname in os.listdir(solution_dir):
        if fname.endswith(".pkl"):
            try:
                with open(os.path.join(solution_dir, fname), "rb") as f:
                    sol = pickle.load(f)
                required_keys = ['params', 'X', 'Y', 'c1_preds', 'c2_preds', 'times']
                if all(key in sol for key in required_keys):
                    if not (
                        np.any(np.isnan(sol['c1_preds'])) or np.any(np.isnan(sol['c2_preds'])) or
                        np.all(sol['c1_preds'] == 0) or np.all(sol['c2_preds'] == 0)
                    ):
                        solutions.append(sol)
                        lys.append(sol['params']['Ly'])
            except Exception:
                continue
    return solutions, sorted(set(lys))


def safe_save(fig, output_dir, base_filename):
    """Safely export Plotly figures (HTML always, PNG optional)."""
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f"{base_filename}.html")
    fig.write_html(html_path)
    try:
        fig.write_image(os.path.join(output_dir, f"{base_filename}.png"), format="png")
    except Exception as e:
        st.warning(f"⚠️ PNG export failed (Kaleido issue likely): {e}")
    return html_path


# ===================================
# Chart functions
# ===================================
def plot_radar_concentration(solution, species="Cu", output_dir="figures"):
    """Radar chart with actual concentration values as labels."""
    Ly = solution['params']['Ly']
    times = solution['times']
    c_all = solution['c1_preds'] if species == "Cu" else solution['c2_preds']

    # Handle shape: (Nt, Ny, Nx) or (1, Nt, Ny, Nx)
    if c_all.ndim == 4:
        c_all = c_all[0]
    if c_all.ndim != 3:
        raise ValueError(f"Unexpected shape for concentration array: {c_all.shape}")

    ny, nx = c_all.shape[1], c_all.shape[2]
    center_x = nx // 2

    # Use actual concentration values as angular labels
    c_axis = c_all.mean(axis=(0, 1)) if ny * nx > 0 else np.linspace(0, 1, ny)
    theta_labels = [f"{c_val:.2f}" for c_val in np.linspace(0, 1, len(c_axis))]

    fig = go.Figure()
    for t_idx, t_val in enumerate(times):
        c_profile = c_all[t_idx, :, center_x]
        fig.add_trace(go.Scatterpolar(
            r=[t_val] * len(c_profile),
            theta=theta_labels,
            mode="markers+text",
            text=[f"{c:.3f}" for c in c_profile],
            textposition="top center",
            name=f"t={t_val:.1f}s",
            hovertemplate="t=%{r:.2f}s<br>c=%{text}<extra></extra>"
        ))

    fig.update_layout(
        title=f"{species} Radar Chart — Time vs Concentration<br>Ly={Ly:.1f} μm",
        polar=dict(
            radialaxis=dict(visible=True, title="Time (s)"),
            angularaxis=dict(direction="clockwise")
        ),
        showlegend=True
    )

    base_filename = f"radar_conc_{species.lower()}_ly_{Ly:.1f}"
    safe_save(fig, output_dir, base_filename)
    return fig, base_filename


def plot_polar_concentration(solution, species="Cu", output_dir="figures"):
    """Polar chart — radius=time, circumference=concentration labels."""
    Ly = solution['params']['Ly']
    times = solution['times']
    c_all = solution['c1_preds'] if species == "Cu" else solution['c2_preds']

    # Handle shape: (Nt, Ny, Nx) or (1, Nt, Ny, Nx)
    if c_all.ndim == 4:
        c_all = c_all[0]
    if c_all.ndim != 3:
        raise ValueError(f"Unexpected shape for concentration array: {c_all.shape}")

    ny, nx = c_all.shape[1], c_all.shape[2]
    center_x = nx // 2
    c_axis = np.linspace(0, 1, ny)
    theta = np.linspace(0, 2 * np.pi, len(c_axis))
    theta_labels = [f"{c_val:.2f}" for c_val in c_axis]

    fig = go.Figure()
    for t_idx, t_val in enumerate(times):
        c_profile = c_all[t_idx, :, center_x]
        fig.add_trace(go.Scatterpolar(
            r=[t_val] * len(c_profile),
            theta=theta * 180 / np.pi,
            mode="markers",
            marker=dict(size=8, color=c_profile, colorscale="Viridis", colorbar_title="Conc."),
            name=f"t={t_val:.1f}s",
            hovertemplate="θ=%{theta:.1f}°<br>c=%{marker.color:.2e}<extra></extra>"
        ))

    fig.update_layout(
        title=f"{species} Polar Chart — Time vs Concentration<br>Ly={Ly:.1f} μm",
        polar=dict(
            radialaxis=dict(visible=True, title="Time (s)"),
            angularaxis=dict(tickvals=np.linspace(0, 360, len(theta_labels)), ticktext=theta_labels)
        ),
        showlegend=False
    )

    base_filename = f"polar_conc_{species.lower()}_ly_{Ly:.1f}"
    safe_save(fig, output_dir, base_filename)
    return fig, base_filename


# ===================================
# Streamlit UI
# ===================================
def main():
    st.title("PINN Solution Visualization")

    # Load data
    solutions, lys = load_solutions(SOLUTION_DIR)
    if not solutions:
        st.error("No valid solution files found in pinn_solutions directory.")
        return

    st.subheader("Select Parameters")
    ly_choices = st.multiselect("Select Domain Height(s) Ly (μm)",
                                options=lys,
                                default=[lys[0]] if lys else [])

    species = st.selectbox("Select Species", ["Cu", "Ni"])
    chart_type = st.selectbox("Select Chart Type", ["Radar Chart", "Polar Chart"])

    for ly in ly_choices:
        # find solution for this Ly
        solution = next((s for s in solutions if abs(s['params']['Ly'] - ly) < 1e-6), None)
        if not solution:
            st.warning(f"No solution found for Ly={ly:.1f} μm")
            continue

        st.markdown(f"### Ly = {ly:.1f} μm — {species}")
        if chart_type == "Radar Chart":
            fig, filename = plot_radar_concentration(solution, species)
        else:
            fig, filename = plot_polar_concentration(solution, species)

        st.plotly_chart(fig, use_container_width=True)

        html_path = os.path.join("figures", f"{filename}.html")
        if os.path.exists(html_path):
            st.download_button(
                label=f"Download {chart_type} ({species})",
                data=open(html_path, "rb").read(),
                file_name=f"{filename}.html",
                mime="text/html"
            )


if __name__ == "__main__":
    main()
