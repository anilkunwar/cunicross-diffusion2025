import os
import pickle
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import plotly.io as pio

# === Kaleido default settings ===
pio.kaleido.scope.default_format = "png"
pio.kaleido.scope.default_width = 1000
pio.kaleido.scope.default_height = 800

SOLUTION_DIR = Path(__file__).parent / "pinn_solutions"

# Fixed boundary conditions
C_CU_TOP = 0.0
C_CU_BOTTOM = 1.6e-3
C_NI_TOP = 1.25e-3
C_NI_BOTTOM = 0.0

# === Load solutions ===
@st.cache_data
def load_solutions(solution_dir):
    solutions = []
    lys = []
    for fname in os.listdir(solution_dir):
        if fname.endswith(".pkl"):
            try:
                with open(os.path.join(solution_dir, fname), "rb") as f:
                    sol = pickle.load(f)
                required_keys = ['params', 'X', 'Y', 'c1_preds', 'c2_preds', 'times']
                if all(key in sol for key in required_keys):
                    if not (np.any(np.isnan(sol['c1_preds'])) or np.any(np.isnan(sol['c2_preds'])) or
                            np.all(sol['c1_preds'] == 0) or np.all(sol['c2_preds'] == 0)):
                        solutions.append(sol)
                        lys.append(sol['params']['Ly'])
            except Exception:
                continue
    return solutions, sorted(set(lys))

def safe_save(fig, output_dir, base_filename):
    os.makedirs(output_dir, exist_ok=True)
    try:
        fig.write_image(os.path.join(output_dir, f"{base_filename}.png"), scale=2)
    except Exception as e:
        st.warning(f"PNG export failed: {e}")
    fig.write_html(os.path.join(output_dir, f"{base_filename}.html"))

# === Utility: get concentration at center ===
def get_center_conc(solution, species="Cu"):
    c_all = solution['c1_preds'] if species == "Cu" else solution['c2_preds']
    center_idx_x = c_all.shape[1] // 2  # Lx/2
    center_idx_y = c_all.shape[2] // 2  # Ly/2
    return [c_all[t_idx, center_idx_x, center_idx_y] for t_idx in range(len(solution['times']))]

# === Radar chart for Cu concentration ===
def plot_radar_chart_cu(solutions, selected_lys, output_dir="figures"):
    categories = ['y=0 (Bottom)', 'y=Ly/4', 'y=Ly/2 (Center)', 'y=3Ly/4', 'y=Ly (Top)']
    fig = go.Figure()
    
    colors = ['blue', 'red']  # Colors for Ly=30 and Ly=120
    for idx, ly_choice in enumerate(selected_lys):
        solution = next((sol for sol in solutions if abs(sol['params']['Ly'] - ly_choice) < 0.1), None)
        if solution:
            times = solution['times']
            c_all = solution['c1_preds']  # Cu concentrations
            center_idx_x = c_all.shape[1] // 2  # Lx/2
            y_coords = solution['Y'][0, :]
            y_indices = [0, len(y_coords)//4, len(y_coords)//2, 3*len(y_coords)//4, -1]  # Bottom, Ly/4, Center, 3Ly/4, Top
            
            # Sample time points with increased step size
            time_indices = np.linspace(0, len(times)-1, 5, dtype=int)  # 5 time points
            for t_idx in time_indices:
                t_val = times[t_idx]
                conc = [c_all[t_idx, center_idx_x, y_idx] for y_idx in y_indices]
                fig.add_trace(go.Scatterpolar(
                    r=[t_val] * len(categories),  # Time as radial axis
                    theta=categories,
                    mode="markers+lines",
                    marker=dict(size=8, color=conc, colorscale="Viridis", showscale=(idx==0 and t_idx==time_indices[0]), colorbar_title="Cu Conc. (mol/cc)"),
                    line=dict(color=colors[idx], width=2),
                    name=f"Ly={ly_choice:.1f}, t={t_val:.1f}s",
                    hovertemplate=f"Ly={ly_choice:.1f}, t={t_val:.1f}s<br>%{{theta}}: %{{marker.color:.2e}}<extra></extra>"
                ))

    fig.update_layout(
        title="Cu Concentration Evolution at Centerline (x=Lx/2)<br>Time (radial), Position (angular)",
        polar=dict(
            radialaxis=dict(visible=True, title="Time (s)"),
            angularaxis=dict(visible=True)
        ),
        showlegend=True,
        font=dict(size=12),
        margin=dict(l=50, r=50, t=100, b=50),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    
    base_filename = f"radar_cu_lys_{'_'.join([str(ly) for ly in selected_lys])}"
    safe_save(fig, output_dir, base_filename)
    return fig, base_filename

# === Radar chart for Ni concentration ===
def plot_radar_chart_ni(solutions, selected_lys, output_dir="figures"):
    categories = ['y=0 (Bottom)', 'y=Ly/4', 'y=Ly/2 (Center)', 'y=3Ly/4', 'y=Ly (Top)']
    fig = go.Figure()
    
    colors = ['blue', 'red']  # Colors for Ly=30 and Ly=120
    for idx, ly_choice in enumerate(selected_lys):
        solution = next((sol for sol in solutions if abs(sol['params']['Ly'] - ly_choice) < 0.1), None)
        if solution:
            times = solution['times']
            c_all = solution['c2_preds']  # Ni concentrations
            center_idx_x = c_all.shape[1] // 2  # Lx/2
            y_coords = solution['Y'][0, :]
            y_indices = [0, len(y_coords)//4, len(y_coords)//2, 3*len(y_coords)//4, -1]  # Bottom, Ly/4, Center, 3Ly/4, Top
            
            # Sample time points with increased step size
            time_indices = np.linspace(0, len(times)-1, 5, dtype=int)  # 5 time points
            for t_idx in time_indices:
                t_val = times[t_idx]
                conc = [c_all[t_idx, center_idx_x, y_idx] for y_idx in y_indices]
                fig.add_trace(go.Scatterpolar(
                    r=[t_val] * len(categories),  # Time as radial axis
                    theta=categories,
                    mode="markers+lines",
                    marker=dict(size=8, color=conc, colorscale="Magma", showscale=(idx==0 and t_idx==time_indices[0]), colorbar_title="Ni Conc. (mol/cc)"),
                    line=dict(color=colors[idx], width=2),
                    name=f"Ly={ly_choice:.1f}, t={t_val:.1f}s",
                    hovertemplate=f"Ly={ly_choice:.1f}, t={t_val:.1f}s<br>%{{theta}}: %{{marker.color:.2e}}<extra></extra>"
                ))

    fig.update_layout(
        title="Ni Concentration Evolution at Centerline (x=Lx/2)<br>Time (radial), Position (angular)",
        polar=dict(
            radialaxis=dict(visible=True, title="Time (s)"),
            angularaxis=dict(visible=True)
        ),
        showlegend=True,
        font=dict(size=12),
        margin=dict(l=50, r=50, t=100, b=50),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    
    base_filename = f"radar_ni_lys_{'_'.join([str(ly) for ly in selected_lys])}"
    safe_save(fig, output_dir, base_filename)
    return fig, base_filename

# === Streamlit App ===
def main():
    st.title("🔬 PINN Cu and Ni Concentration Evolution at Centerline")

    # Load solutions
    solutions, lys = load_solutions(SOLUTION_DIR)
    if not solutions:
        st.error("No valid solutions found.")
        return

    # Filter for Ly=30 and Ly=120
    selected_lys = [ly for ly in [30.0, 120.0] if ly in lys]
    if not selected_lys:
        st.error("Solutions for Ly=30 or Ly=120 not found.")
        return

    st.subheader("Cu Concentration Radar Chart")
    fig_cu, filename_cu = plot_radar_chart_cu(solutions, selected_lys)
    st.plotly_chart(fig_cu, use_container_width=True)

    # Download buttons for Cu chart
    html_path_cu = os.path.join("figures", f"{filename_cu}.html")
    if os.path.exists(html_path_cu):
        st.download_button(
            "⬇️ Download Cu Radar Chart (HTML)",
            data=open(html_path_cu, "rb").read(),
            file_name=f"{filename_cu}.html",
            mime="text/html"
        )
    png_path_cu = os.path.join("figures", f"{filename_cu}.png")
    if os.path.exists(png_path_cu):
        st.download_button(
            "⬇️ Download Cu Radar Chart (PNG)",
            data=open(png_path_cu, "rb").read(),
            file_name=f"{filename_cu}.png",
            mime="image/png"
        )
    else:
        st.info("PNG export unavailable (Kaleido not installed).")

    st.subheader("Ni Concentration Radar Chart")
    fig_ni, filename_ni = plot_radar_chart_ni(solutions, selected_lys)
    st.plotly_chart(fig_ni, use_container_width=True)

    # Download buttons for Ni chart
    html_path_ni = os.path.join("figures", f"{filename_ni}.html")
    if os.path.exists(html_path_ni):
        st.download_button(
            "⬇️ Download Ni Radar Chart (HTML)",
            data=open(html_path_ni, "rb").read(),
            file_name=f"{filename_ni}.html",
            mime="text/html"
        )
    png_path_ni = os.path.join("figures", f"{filename_ni}.png")
    if os.path.exists(png_path_ni):
        st.download_button(
            "⬇️ Download Ni Radar Chart (PNG)",
            data=open(png_path_ni, "rb").read(),
            file_name=f"{filename_ni}.png",
            mime="image/png"
        )
    else:
        st.info("PNG export unavailable (Kaleido not installed).")

if __name__ == "__main__":
    main()
