import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# === Utility for saving charts ===
def safe_save(fig, output_dir, base_filename):
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f"{base_filename}.html")
    fig.write_html(html_path)
    try:
        png_path = os.path.join(output_dir, f"{base_filename}.png")
        fig.write_image(png_path, width=1400, height=1000, scale=2)
    except Exception as e:
        st.warning(f"⚠️ PNG export failed (Kaleido issue): {e}")
        png_path = None
    return html_path, png_path

# === Radar / Polar Chart ===
def plot_radar_concentration(solution, species="Cu",
                             r_tick_step=None, theta_tick_step=None,
                             show_radial_labels=True, show_grid=True,
                             font_size=16, colorscale="Viridis",
                             output_dir="figures"):

    Ly = solution["params"]["Ly"]
    times = np.array(solution["times"])
    c_all = solution["c1_preds"] if species=="Cu" else solution["c2_preds"]

    if c_all.ndim == 4:
        c_all = c_all[0]

    nt, ny, nx = c_all.shape
    center_x = nx // 2
    conc_profiles = c_all[:, :, center_x]  # shape: (nt, ny)
    theta_deg = np.linspace(0, 360, ny)

    fig = go.Figure()

    # Use time as color for marker
    for t_idx, t_val in enumerate(times):
        conc = conc_profiles[t_idx]
        fig.add_trace(go.Scatterpolar(
            r=[t_val] * ny,
            theta=theta_deg,
            mode="markers+lines",
            marker=dict(
                size=10,
                color=conc,   # color = concentration
                colorscale=colorscale,
                cmin=conc_profiles.min(),
                cmax=conc_profiles.max(),
                colorbar=dict(
                    title=f"{species} Conc (mol/cc)",
                    thickness=25, len=0.6, x=1.1, y=0.5,  # horizontal offset
                    tickfont=dict(size=font_size)
                )
            ),
            line=dict(color="gray", width=1),
            name=f"t={t_val:.2f}s",
            hovertemplate="t=%{r:.2f}s<br>Conc=%{marker.color:.2e}<extra></extra>"
        ))

    # Radial & angular ticks
    radial_ticks = np.arange(times.min(), times.max() + (r_tick_step or 1), r_tick_step or 1)
    if theta_tick_step:
        angular_ticks = np.arange(0, 360, theta_tick_step)
        indices = np.clip((angular_ticks / 360 * ny).astype(int), 0, ny-1)
        angular_labels = [f"{conc_profiles[-1, i]:.2e}" for i in indices]
    else:
        angular_ticks = None
        angular_labels = None

    fig.update_layout(
        title=f"{species} Radar Chart — Ly={Ly:.2e} m",
        polar=dict(
            radialaxis=dict(visible=show_radial_labels, tickvals=radial_ticks,
                            showgrid=show_grid, title="Time (s)" if show_radial_labels else None,
                            tickfont=dict(size=font_size)),
            angularaxis=dict(tickvals=angular_ticks, ticktext=angular_labels,
                             showgrid=show_grid, tickfont=dict(size=font_size))
        ),
        template="plotly_white",
        font=dict(size=font_size),
        margin=dict(l=120, r=180, t=120, b=120),
        showlegend=True
    )

    base_filename = f"radar_conc_{species.lower()}_ly_{Ly:.2e}"
    safe_save(fig, output_dir, base_filename)
    return fig

# === Streamlit App ===
def main():
    st.title("📊 Cu-Ni Diffusion Radar/Polar Charts — Enhanced Publication Quality")

    # Demo solution (replace with your PINN solution)
    times = np.linspace(0, 10, 6)
    nx, ny = 40, 40
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    Cu_data = np.array([np.exp(-Y*(0.5+0.05*t))*2.85e-3 for t in times])
    Ni_data = np.array([np.exp(-Y*(0.3+0.03*t))*1.75e-3 for t in times])
    solution = {"params":{"Ly":2.85e-3}, "times":times, "c1_preds":Cu_data, "c2_preds":Ni_data}

    # User controls
    species = st.selectbox("Select Species:", ["Cu","Ni"])
    r_step = st.number_input("Radial tick step (Time)", value=2.0, step=0.5)
    theta_step = st.number_input("Angular tick step (Positions)", value=45.0, step=5.0)
    font_size = st.slider("Font size", 10, 30, 16)
    show_radial_labels = st.checkbox("Show Radial (Time) Labels", value=True)
    show_grid = st.checkbox("Show Grid", value=True)
    color_options = list(px.colors.named_colorscales())
    default_index = color_options.index("Viridis") if "Viridis" in color_options else 0
    colorscale = st.selectbox("Select Colorscale", color_options, index=default_index)

    st.subheader("Radar Chart")
    fig_radar = plot_radar_concentration(solution, species=species,
                                         r_tick_step=r_step, theta_tick_step=theta_step,
                                         show_radial_labels=show_radial_labels, show_grid=show_grid,
                                         font_size=font_size, colorscale=colorscale)
    st.plotly_chart(fig_radar, use_container_width=True)

if __name__=="__main__":
    main()
