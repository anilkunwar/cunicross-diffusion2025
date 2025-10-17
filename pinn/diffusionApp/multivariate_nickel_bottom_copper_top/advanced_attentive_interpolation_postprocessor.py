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
        fig.write_image(png_path, width=1200, height=900, scale=2)
    except Exception as e:
        st.warning(f"⚠️ PNG export failed (Kaleido issue): {e}")
        png_path = None
    return html_path, png_path

# === Radar chart with time colorbar ===
def plot_radar_time_colors(solution, species="Cu",
                           show_radial_labels=True, colorscale="Viridis",
                           output_dir="figures"):

    Ly = solution["params"]["Ly"]
    times = np.array(solution["times"])
    c_all = solution["c1_preds"] if species=="Cu" else solution["c2_preds"]

    if c_all.ndim == 4:
        c_all = c_all[0]  # remove batch dim if exists

    nt, ny, nx = c_all.shape
    center_x = nx // 2
    conc_profiles = c_all[:, :, center_x]  # shape: (nt, ny)
    theta_deg = np.linspace(0, 360, ny)

    fig = go.Figure()

    for t_idx, t_val in enumerate(times):
        conc = conc_profiles[t_idx]
        fig.add_trace(go.Scatterpolar(
            r=theta_deg,                   # radial = angular position
            theta=[t_val]*ny,              # angular = time (optional, just to spread)
            mode="markers+lines",
            marker=dict(
                size=10,
                color=t_val,               # time encoded in color
                colorscale=colorscale,
                cmin=times.min(),
                cmax=times.max(),
                colorbar=dict(title="Time (s)", thickness=20, len=0.5, y=0.5)
            ),
            line=dict(color="gray", width=1),
            name=f"t={t_val:.2f}s",
            hovertemplate="Position=%{r:.2f}<br>Conc=%{marker.color:.2e}<extra></extra>"
        ))

    fig.update_layout(
        title=f"{species} Radar Chart — Ly={Ly:.2e} m (Time colorbar)",
        polar=dict(
            radialaxis=dict(visible=show_radial_labels, title="Position (deg)"),
            angularaxis=dict(visible=False)
        ),
        template="plotly_white",
        font=dict(size=16),
        margin=dict(l=120, r=120, t=120, b=120),
        showlegend=True
    )

    base_filename = f"radar_time_colors_{species.lower()}_ly_{Ly:.2e}"
    safe_save(fig, output_dir, base_filename)
    return fig
