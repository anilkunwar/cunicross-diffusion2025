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
        st.warning(f"⚠️ PNG export failed: {e}")
    fig.write_html(os.path.join(output_dir, f"{base_filename}.html"))

# === Utility: get concentration at center ===
def get_center_conc(solution, species="Cu"):
    c_all = solution['c1_preds'] if species == "Cu" else solution['c2_preds']
    center_idx_x = c_all.shape[1] // 2  # Lx/2
    center_idx_y = c_all.shape[2] // 2  # Ly/2
    return [c_all[t_idx, center_idx_x, center_idx_y] for t_idx in range(len(solution['times']))]

# === Polar chart (concentration as radial, time as theta, center point) ===
def plot_polar_chart_center(solutions, selected_lys, species="Cu", output_dir="figures"):
    fig = go.Figure()
    for ly_choice in selected_lys:
        solution = next((sol for sol in solutions if abs(sol['params']['Ly'] - ly_choice) < 0.1), None)
        if solution:
            times = solution['times']
            conc = get_center_conc(solution, species)
            theta = np.linspace(0, 2 * np.pi, len(times))
            
            fig.add_trace(go.Scatterpolar(
                r=conc,
                theta=theta * 180 / np.pi,
                mode="lines+markers",
                name=f"Ly={ly_choice:.1f}",
                marker=dict(size=6, symbol="circle"),
                hovertemplate="Time=%{theta:.1f}°, Conc=%{r:.2e}<extra></extra>"
            ))

    fig.update_layout(
        title=f"{species} Concentration Evolution at Center (Lx/2=30 μm)<br>Concentration (radial), Time (angular)",
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
                tickvals=[0, 90, 180, 270, 360],
                ticktext=[f"{t:.1f}s" for t in [times[0], times[len(times)//4], times[len(times)//2], times[3*len(times)//4], times[-1]]]
            )
        ),
        showlegend=True,
        font=dict(size=12),
        margin=dict(l=50, r=50, t=100, b=50),
    )
    
    base_filename = f"polar_center_{species.lower()}_lys_{'_'.join([str(ly) for ly in selected_lys])}"
    safe_save(fig, output_dir, base_filename)
    return fig, base_filename

# === Streamlit App ===
def main():
    st.title("🔬 PINN Center Concentration Evolution")

    # Load solutions
    solutions, lys = load_solutions(SOLUTION_DIR)
    if not solutions:
        return

    # Ly selection (multiple)
    st.subheader("Select Domain Heights (Ly)")
    selected_lys = st.multiselect(
        "Select Ly values to compare (up to 4)",
        options=lys,
        default=lys[:min(4, len(lys))],
        format_func=lambda x: f"{x:.1f} μm"
    )

    # Species selection
    species = st.selectbox("Select Species", ["Cu", "Ni"])

    if selected_lys:
        st.subheader(f"{species} Concentration at Center (Lx/2=30 μm, Ly/2)")
        fig, filename = plot_polar_chart_center(solutions, selected_lys, species=species)
        st.plotly_chart(fig, use_container_width=True)

        # Download buttons
        html_path = os.path.join("figures", f"{filename}.html")
        if os.path.exists(html_path):
            st.download_button(
                "⬇️ Download Polar Chart (HTML)",
                data=open(html_path, "rb").read(),
                file_name=f"{filename}.html",
                mime="text/html"
            )
        png_path = os.path.join("figures", f"{filename}.png")
        if os.path.exists(png_path):
            st.download_button(
                "⬇️ Download Polar Chart (PNG)",
                data=open(png_path, "rb").read(),
                file_name=f"{filename}.png",
                mime="image/png"
            )
        else:
            st.info("PNG export unavailable (Kaleido not installed).")

if __name__ == "__main__":
    main()
