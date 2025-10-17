import os
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# === Utility for saving charts ===
def safe_save(fig, output_dir, base_filename):
    os.makedirs(output_dir, exist_ok=True)
    # Save as HTML and attempt PNG (high-resolution)
    html_path = os.path.join(output_dir, f"{base_filename}.html")
    fig.write_html(html_path)
    try:
        png_path = os.path.join(output_dir, f"{base_filename}.png")
        fig.write_image(png_path, width=1200, height=900, scale=2)
    except Exception as e:
        st.warning(f"⚠️ PNG export failed (Kaleido issue): {e}")
        png_path = None
    return html_path, png_path

# === Radar chart: concentration vs time ===
def plot_radar_concentration(solution, species="Cu", r_tick_step=None, theta_tick_step=None, output_dir="figures"):
    Ly = solution["params"]["Ly"]
    times = np.array(solution["times"])
    c_all = solution["c1_preds"] if species == "Cu" else solution["c2_preds"]

    if c_all.ndim == 4:  # handle extra batch dimension
        c_all = c_all[0]
    nt, ny, nx = c_all.shape
    center_x = nx // 2

    # Concentration values along y-axis
    conc_profiles = c_all[:, :, center_x]  # shape: (nt, ny)

    # Angular axis: y positions
    theta_deg = np.linspace(0, 360, ny)
    theta_labels = [f"{c:.2e}" for c in conc_profiles[-1]]  # last time snapshot for labels

    fig = go.Figure()

    # Color scale representation: average across angular positions
    for t_idx, t_val in enumerate(times):
        conc = conc_profiles[t_idx]
        fig.add_trace(
            go.Scatterpolar(
                r=[t_val]*ny,
                theta=theta_deg,
                mode="markers+lines",
                marker=dict(
                    size=8,
                    color=conc,
                    colorscale="Viridis" if species=="Cu" else "Magma",
                    colorbar=dict(title=f"{species} Conc (mol/cc)"),
                    cmin=conc_profiles.min(),
                    cmax=conc_profiles.max(),
                ),
                line=dict(color="gray", width=1),
                name=f"t={t_val:.2f}s",
                hovertemplate="t=%{r:.2f}s<br>Conc=%{marker.color:.2e}<extra></extra>"
            )
        )

    # Tick control
    radial_ticks = None
    angular_ticks = None
    if r_tick_step:
        radial_ticks = np.arange(times.min(), times.max() + r_tick_step, r_tick_step)
    if theta_tick_step:
        angular_ticks = np.arange(0, 360 + theta_tick_step, theta_tick_step)

    fig.update_layout(
        title=f"{species} Radar Chart — Ly={Ly:.2e} m",
        polar=dict(
            radialaxis=dict(visible=True, title="Time (s)", tickvals=radial_ticks),
            angularaxis=dict(tickvals=angular_ticks, ticktext=theta_labels if angular_ticks else theta_labels)
        ),
        template="plotly_white",
        font=dict(size=14),
        margin=dict(l=80, r=80, t=120, b=80),
        showlegend=True
    )

    base_filename = f"radar_conc_{species.lower()}_ly_{Ly:.2e}"
    safe_save(fig, output_dir, base_filename)
    return fig

# === Polar chart: alternative view ===
def plot_polar_concentration(solution, species="Cu", r_tick_step=None, theta_tick_step=None, output_dir="figures"):
    Ly = solution["params"]["Ly"]
    times = np.array(solution["times"])
    c_all = solution["c1_preds"] if species == "Cu" else solution["c2_preds"]

    if c_all.ndim == 4:
        c_all = c_all[0]
    nt, ny, nx = c_all.shape
    center_x = nx // 2

    theta_deg = np.linspace(0, 360, ny)
    theta_labels = [f"{c:.2e}" for c in c_all[-1, :, center_x]]

    fig = go.Figure()
    for t_idx, t_val in enumerate(times):
        conc = c_all[t_idx, :, center_x]
        fig.add_trace(
            go.Scatterpolar(
                r=[t_val]*ny,
                theta=theta_deg,
                mode="markers",
                marker=dict(
                    size=8,
                    color=conc,
                    colorscale="Viridis" if species=="Cu" else "Magma",
                    cmin=c_all.min(),
                    cmax=c_all.max(),
                    colorbar=dict(title=f"{species} Conc (mol/cc)")
                ),
                name=f"t={t_val:.2f}s"
            )
        )

    radial_ticks = np.arange(times.min(), times.max() + (r_tick_step or 1), r_tick_step or 1)
    angular_ticks = np.arange(0, 360 + (theta_tick_step or 10), theta_tick_step or 10)

    fig.update_layout(
        title=f"{species} Polar Chart — Ly={Ly:.2e} m",
        polar=dict(
            radialaxis=dict(visible=True, title="Time (s)", tickvals=radial_ticks),
            angularaxis=dict(tickvals=angular_ticks, ticktext=theta_labels)
        ),
        template="plotly_white",
        font=dict(size=14),
        margin=dict(l=80, r=80, t=120, b=80),
        showlegend=False
    )

    base_filename = f"polar_conc_{species.lower()}_ly_{Ly:.2e}"
    safe_save(fig, output_dir, base_filename)
    return fig

# === Streamlit app ===
def main():
    st.title("📊 Publication-Quality Cu-Ni Diffusion Radar/Polar Charts")

    # Demo solution (replace with your loaded solution)
    times = np.linspace(0, 10, 6)
    nx, ny = 40, 40
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    Cu_data = np.array([np.exp(-Y*(0.5+0.05*t))*2.85e-3 for t in times])
    Ni_data = np.array([np.exp(-Y*(0.3+0.03*t))*1.75e-3 for t in times])
    solution = {"params": {"Ly": 2.85e-3}, "times": times, "c1_preds": Cu_data, "c2_preds": Ni_data}

    species = st.selectbox("Species:", ["Cu", "Ni"])
    r_step = st.number_input("Radial tick step (Time)", value=2.0, step=0.5)
    theta_step = st.number_input("Angular tick step (positions)", value=45.0, step=5.0)

    st.subheader("Radar Chart")
    fig_radar = plot_radar_concentration(solution, species=species, r_tick_step=r_step, theta_tick_step=theta_step)
    st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader("Polar Chart")
    fig_polar = plot_polar_concentration(solution, species=species, r_tick_step=r_step, theta_tick_step=theta_step)
    st.plotly_chart(fig_polar, use_container_width=True)


if __name__ == "__main__":
    main()
