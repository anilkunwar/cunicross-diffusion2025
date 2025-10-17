import os
import numpy as np
import plotly.graph_objects as go
import streamlit as st


# === Utility for safe saving ===
def safe_save(fig, output_dir, base_filename):
    os.makedirs(output_dir, exist_ok=True)
    try:
        filename = os.path.join(output_dir, f"{base_filename}.png")
        fig.write_image(filename)
    except Exception as e:
        st.warning(f"⚠️ PNG export failed: {e}")
        filename = os.path.join(output_dir, f"{base_filename}.html")
        fig.write_html(filename)
    return filename


# === Plot radar chart (actual concentration labels on circumference) ===
def plot_radar_concentration(solution, species="Cu", output_dir="figures"):
    Ly = solution["params"]["Ly"]
    times = np.array(solution["times"])
    c_all = solution["c1_preds"] if species == "Cu" else solution["c2_preds"]

    # Handle shape automatically
    if c_all.ndim == 4:
        c_all = c_all[0]
    if c_all.ndim != 3:
        raise ValueError(f"Unexpected shape for concentration array: {c_all.shape}")

    nt, ny, nx = c_all.shape
    center_x = nx // 2

    # Use real concentration values along y-axis
    c_axis = c_all[-1, :, center_x]  # final time snapshot
    theta_labels = [f"{c_val:.2e}" for c_val in c_axis]
    theta_deg = np.linspace(0, 360, len(c_axis))

    fig = go.Figure()

    for t_idx, t_val in enumerate(times):
        c_profile = c_all[t_idx, :, center_x]
        fig.add_trace(
            go.Scatterpolar(
                r=[t_val] * len(c_profile),
                theta=theta_deg,
                mode="markers+text",
                text=[f"{c:.2e}" for c in c_profile],
                textposition="top center",
                name=f"t={t_val:.2f} s",
                hovertemplate="t=%{r:.2f}s<br>c=%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"{species} Radar Chart — Time vs Concentration<br>Ly = {Ly:.2e} mol/cc",
        polar=dict(
            radialaxis=dict(visible=True, title="Time (s)", showline=True),
            angularaxis=dict(
                tickvals=theta_deg, ticktext=theta_labels, direction="clockwise"
            ),
        ),
        showlegend=True,
        template="plotly_dark",
    )

    base_filename = f"radar_conc_{species.lower()}_ly_{Ly:.2e}"
    safe_save(fig, output_dir, base_filename)
    return fig, base_filename


# === Polar chart (colored by concentration, radial=time, circumference=concentration) ===
def plot_polar_concentration(solution, species="Cu", output_dir="figures"):
    Ly = solution["params"]["Ly"]
    times = np.array(solution["times"])
    c_all = solution["c1_preds"] if species == "Cu" else solution["c2_preds"]

    if c_all.ndim == 4:
        c_all = c_all[0]
    if c_all.ndim != 3:
        raise ValueError(f"Unexpected shape for concentration array: {c_all.shape}")

    nt, ny, nx = c_all.shape
    center_x = nx // 2

    c_axis = c_all[-1, :, center_x]
    theta_deg = np.linspace(0, 360, len(c_axis))
    theta_labels = [f"{c_val:.2e}" for c_val in c_axis]

    fig = go.Figure()
    for t_idx, t_val in enumerate(times):
        c_profile = c_all[t_idx, :, center_x]
        fig.add_trace(
            go.Scatterpolar(
                r=[t_val] * len(c_profile),
                theta=theta_deg,
                mode="markers",
                marker=dict(
                    size=8,
                    color=c_profile,
                    colorscale="Viridis",
                    colorbar_title="Conc (mol/cc)",
                ),
                name=f"t={t_val:.2f}s",
            )
        )

    fig.update_layout(
        title=f"{species} Polar Chart — Time vs Concentration<br>Ly = {Ly:.2e} mol/cc",
        polar=dict(
            radialaxis=dict(visible=True, title="Time (s)"),
            angularaxis=dict(
                tickvals=theta_deg, ticktext=theta_labels, direction="clockwise"
            ),
        ),
        showlegend=False,
        template="plotly_dark",
    )

    base_filename = f"polar_conc_{species.lower()}_ly_{Ly:.2e}"
    safe_save(fig, output_dir, base_filename)
    return fig, base_filename


# === Streamlit app ===
def main():
    st.title("🧩 Cu-Ni Diffusion — Polar and Radar Visualization")

    # Example solution structure (replace with loaded data)
    times = np.linspace(0, 10, 6)
    nx, ny = 40, 40
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    # Fake data for demonstration (Cu higher on top, Ni lower)
    Cu_data = np.array(
        [np.exp(-Y * (0.5 + 0.05 * t)) for t in times]
    ) * 2.85e-3  # mol/cc
    Ni_data = np.array(
        [np.exp(-Y * (0.3 + 0.03 * t)) for t in times]
    ) * 1.75e-3  # mol/cc

    # Two Ly configurations for comparison
    solution_L1 = {"params": {"Ly": 2.85e-3}, "times": times, "c1_preds": Cu_data, "c2_preds": Ni_data}
    solution_L2 = {"params": {"Ly": 3.25e-3}, "times": times, "c1_preds": Cu_data * 0.9, "c2_preds": Ni_data * 1.1}

    species_choice = st.radio("Select species:", ["Cu", "Ni"])
    compare_mode = st.checkbox("Compare two Ly values")

    if compare_mode:
        col1, col2 = st.columns(2)
        with col1:
            fig1, _ = plot_radar_concentration(solution_L1, species_choice)
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2, _ = plot_radar_concentration(solution_L2, species_choice)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        fig, _ = plot_radar_concentration(solution_L1, species_choice)
        st.plotly_chart(fig, use_container_width=True)

    # Add also polar chart below
    st.divider()
    st.subheader("Polar Chart View")

    if compare_mode:
        col1, col2 = st.columns(2)
        with col1:
            fig1, _ = plot_polar_concentration(solution_L1, species_choice)
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2, _ = plot_polar_concentration(solution_L2, species_choice)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        fig, _ = plot_polar_concentration(solution_L1, species_choice)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
