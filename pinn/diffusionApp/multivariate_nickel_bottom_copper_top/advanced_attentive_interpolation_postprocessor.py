import os
import pickle
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import plotly.io as pio

# === Optional: Kaleido default settings ===
pio.kaleido.scope.default_format = "png"
pio.kaleido.scope.default_width = 800
pio.kaleido.scope.default_height = 600

SOLUTION_DIR = Path(__file__).parent / "pinn_solutions"

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
        fig.write_image(os.path.join(output_dir, f"{base_filename}.png"))
    except Exception as e:
        st.warning(f"⚠️ PNG export failed (Kaleido issue likely): {e}")
    fig.write_html(os.path.join(output_dir, f"{base_filename}.html"))

# === Polar chart (concentration as radial axis, center point only) ===
def plot_polar_chart_center(solution, output_dir="figures"):
    Ly = solution['params']['Ly']
    times = solution['times']
    c1_all = solution['c1_preds']  # Cu concentrations
    c2_all = solution['c2_preds']  # Ni concentrations
    center_idx_x = c1_all.shape[1] // 2  # Lx/2
    center_idx_y = c1_all.shape[2] // 2  # Ly/2

    # Extract concentrations at center point (Lx/2, Ly/2) over time
    cu_conc = [c1_all[t_idx, center_idx_x, center_idx_y] for t_idx in range(len(times))]
    ni_conc = [c2_all[t_idx, center_idx_x, center_idx_y] for t_idx in range(len(times))]
    
    # Use time as angular coordinate
    theta = np.linspace(0, 2 * np.pi, len(times))

    fig = go.Figure()
    # Cu concentration as radial axis
    fig.add_trace(go.Scatterpolar(
        r=cu_conc,
        theta=theta * 180 / np.pi,
        mode="lines+markers",
        name="Cu",
        line=dict(color="blue"),
        marker=dict(size=6),
        hovertemplate="Time=%{theta:.1f}°, Cu Conc=%{r:.2e}<extra></extra>"
    ))
    # Ni concentration as radial axis
    fig.add_trace(go.Scatterpolar(
        r=ni_conc,
        theta=theta * 180 / np.pi,
        mode="lines+markers",
        name="Ni",
        line=dict(color="red"),
        marker=dict(size=6),
        hovertemplate="Time=%{theta:.1f}°, Ni Conc=%{r:.2e}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"Concentration Evolution at Center (Lx/2, Ly/2)<br>Ly={Ly:.1f} μm, Lx=60 μm",
        polar=dict(
            radialaxis=dict(
                visible=True,
                title="Concentration (mol/cc)",
                tickformat=".2e"
            ),
            angularaxis=dict(
                visible=True,
                rotation=90,
                direction="counterclockwise",
                tickvals=[0, 90, 180, 270],
                ticktext=[f"{t:.1f}s" for t in [times[0], times[len(times)//4], times[len(times)//2], times[-1]]]
            )
        ),
        showlegend=True
    )
    
    base_filename = f"polar_center_ly_{Ly:.1f}"
    safe_save(fig, output_dir, base_filename)
    return fig, base_filename

# === Streamlit App ===
def main():
    st.title("🔬 PINN Concentration Evolution at Center (Lx/2, Ly/2)")

    # Load solutions
    solutions, lys = load_solutions(SOLUTION_DIR)
    if not solutions:
        st.error("No valid solutions found.")
        return

    # Ly selection
    st.subheader("Select Domain Height (Ly)")
    ly_choice = st.selectbox(
        "Select Ly value",
        options=lys,
        format_func=lambda x: f"{x:.1f} μm"
    )

    # Find solution
    solution = next((sol for sol in solutions if abs(sol['params']['Ly'] - ly_choice) < 0.1), None)
    if not solution:
        st.warning(f"No solution found for Ly = {ly_choice:.1f}")
        return

    # Generate and display polar chart
    st.subheader(f"Concentration Evolution at Center (Lx/2 = 30 μm, Ly/2 = {ly_choice/2:.1f} μm)")
    fig, filename = plot_polar_chart_center(solution)
    st.plotly_chart(fig, use_container_width=True)

    # Download buttons
    html_path = os.path.join("figures", f"{filename}.html")
    if os.path.exists(html_path):
        st.download_button(
            f"⬇️ Download Polar Chart (HTML)",
            data=open(html_path, "rb").read(),
            file_name=f"{filename}.html",
            mime="text/html"
        )
    png_path = os.path.join("figures", f"{filename}.png")
    if os.path.exists(png_path):
        st.download_button(
            f"⬇️ Download Polar Chart (PNG)",
            data=open(png_path, "rb").read(),
            file_name=f"{filename}.png",
            mime="image/png"
        )
    else:
        st.info("PNG export unavailable (Chrome/Kaleido not installed).")

if __name__ == "__main__":
    main()
