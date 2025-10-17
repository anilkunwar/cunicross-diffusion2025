import os
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# === Utility for saving charts ===
def safe_save(fig, output_dir, base_filename):
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f"{base_filename}.html")
    fig.write_html(html_path)
    try:
        png_path = os.path.join(output_dir, f"{base_filename}.png")
        fig.write_image(png_path, width=1200, height=900, scale=2)
    except Exception as e:
        st.warning(f"⚠️ PNG export failed (Kaleido issue): {e}")
        png_path = None
    return html_path, png_path

# === Radar chart: concentration vs time with enhancements ===
def plot_radar_concentration(solution, species="Cu", r_tick_step=None, theta_tick_step=None,
                             font_size=14, show_grid=True, show_theta_labels=True, output_dir="figures"):
    Ly = solution["params"]["Ly"]
    times = np.array(solution["times"])
    c_all = solution["c1_preds"] if species == "Cu" else solution["c2_preds"]

    if c_all.ndim == 4:
        c_all = c_all[0]
    nt, ny, nx = c_all.shape
    center_x = nx // 2

    conc_profiles = c_all[:, :, center_x]  # shape: (nt, ny)
    theta_deg = np.linspace(0, 360, ny)

    fig = go.Figure()
    # Plot traces for all time points
    for t_idx, t_val in enumerate(times):
        conc = conc_profiles[t_idx]
        fig.add_trace(
            go.Scatterpolar(
                r=[t_val] * ny,
                theta=theta_deg,
                mode="markers+lines",
                marker=dict(
                    size=8,
                    color=conc,
                    colorscale="Viridis" if species=="Cu" else "Magma",
                    cmin=conc_profiles.min()*0.95,  # padding for color scale
                    cmax=conc_profiles.max()*1.05,
                    colorbar=dict(title=f"{species} Conc (mol/cc)", tickfont=dict(size=font_size))
                ),
                line=dict(color="gray", width=1),
                name=f"t={t_val:.2f}s",
                hovertemplate="t=%{r:.2f}s<br>Conc=%{marker.color:.2e}<extra></extra>"
            )
        )

    radial_ticks = np.arange(times.min(), times.max() + (r_tick_step or 1), r_tick_step or 1)
    if theta_tick_step:
        angular_ticks = np.arange(0, 360, theta_tick_step)
        indices = np.clip((angular_ticks / 360 * ny).astype(int), 0, ny-1)
        angular_labels = [f"{conc_profiles[-1, i]:.2e}" for i in indices] if show_theta_labels else ['']*len(indices)
    else:
        angular_ticks = None
        angular_labels = None

    fig.update_layout(
        title=f"{species} Radar Chart — Ly={Ly:.2e} m",
        polar=dict(
            radialaxis=dict(visible=True, title="Time (s)", tickvals=radial_ticks, showgrid=show_grid),
            angularaxis=dict(tickvals=angular_ticks, ticktext=angular_labels, showgrid=show_grid)
        ),
        template="plotly_white",
        font=dict(size=font_size),
        margin=dict(l=80, r=80, t=120, b=80),
        showlegend=True
    )

    base_filename = f"radar_conc_{species.lower()}_ly_{Ly:.2e}"
    safe_save(fig, output_dir, base_filename)
    return fig

# === Polar chart: same enhancements ===
def plot_polar_concentration(solution, species="Cu", r_tick_step=None, theta_tick_step=None,
                             font_size=14, show_grid=True, show_theta_labels=True, output_dir="figures"):
    Ly = solution["params"]["Ly"]
    times = np.array(solution["times"])
    c_all = solution["c1_preds"] if species == "Cu" else solution["c2_preds"]

    if c_all.ndim == 4:
        c_all = c_all[0]
    nt, ny, nx = c_all.shape
    center_x = nx // 2

    conc_profiles = c_all[:, :, center_x]
    theta_deg = np.linspace(0, 360, ny)

    fig = go.Figure()
    for t_idx, t_val in enumerate(times):
        conc = conc_profiles[t_idx]
        fig.add_trace(
            go.Scatterpolar(
                r=[t_val] * ny,
                theta=theta_deg,
                mode="markers",
                marker=dict(
                    size=8,
                    color=conc,
                    colorscale="Viridis" if species=="Cu" else "Magma",
                    cmin=conc_profiles.min()*0.95,
                    cmax=conc_profiles.max()*1.05,
                    colorbar=dict(title=f"{species} Conc (mol/cc)", tickfont=dict(size=font_size))
                ),
                name=f"t={t_val:.2f}s"
            )
        )

    radial_ticks = np.arange(times.min(), times.max() + (r_tick_step or 1), r_tick_step or 1)
    if theta_tick_step:
        angular_ticks = np.arange(0, 360, theta_tick_step)
        indices = np.clip((angular_ticks / 360 * ny).astype(int), 0, ny-1)
        angular_labels = [f"{conc_profiles[-1, i]:.2e}" for i in indices] if show_theta_labels else ['']*len(indices)
    else:
        angular_ticks = None
        angular_labels = None

    fig.update_layout(
        title=f"{species} Polar Chart — Ly={Ly:.2e} m",
        polar=dict(
            radialaxis=dict(visible=True, title="Time (s)", tickvals=radial_ticks, showgrid=show_grid),
            angularaxis=dict(tickvals=angular_ticks, ticktext=angular_labels, showgrid=show_grid)
        ),
        template="plotly_white",
        font=dict(size=font_size),
        margin=dict(l=80, r=80, t=120, b=80),
        showlegend=False
    )

    base_filename = f"polar_conc_{species.lower()}_ly_{Ly:.2e}"
    safe_save(fig, output_dir, base_filename)
    return fig

# === Streamlit App ===
def main():
    st.title("📊 Publication-Quality Cu-Ni Diffusion Radar/Polar Charts")

    # Demo solution (replace with your PINN solution)
    times = np.linspace(0, 10, 6)
    nx, ny = 40, 40
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    Cu_data = np.array([np.exp(-Y*(0.5+0.05*t))*2.85e-3 for t in times])
    Ni_data = np.array([np.exp(-Y*(0.3+0.03*t))*1.75e-3 for t in times])
    solution = {"params": {"Ly": 2.85e-3}, "times": times, "c1_preds": Cu_data, "c2_preds": Ni_data}

    species = st.selectbox("Species:", ["Cu", "Ni"])
    r_step = st.number_input("Radial tick step (Time)", value=2.0, step=0.5)
    theta_step = st.number_input("Angular tick step (Positions)", value=45.0, step=5.0)
    font_size = st.slider("Font size", min_value=10, max_value=24, value=14)
    show_grid = st.checkbox("Show grid", value=True)
    show_labels = st.checkbox("Show angular labels", value=True)

    st.subheader("Radar Chart")
    fig_radar = plot_radar_concentration(solution, species=species, r_tick_step=r_step, theta_tick_step=theta_step,
                                         font_size=font_size, show_grid=show_grid, show_theta_labels=show_labels)
    st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader("Polar Chart")
    fig_polar = plot_polar_concentration(solution, species=species, r_tick_step=r_step, theta_tick_step=theta_step,
                                         font_size=font_size, show_grid=show_grid, show_theta_labels=show_labels)
    st.plotly_chart(fig_polar, use_container_width=True)

if __name__ == "__main__":
    main()
