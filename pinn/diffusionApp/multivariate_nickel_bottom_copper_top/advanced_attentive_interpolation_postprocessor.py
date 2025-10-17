import os
import pickle
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import plotly.io as pio

# === Kaleido settings ===
pio.kaleido.scope.default_format = "png"
pio.kaleido.scope.default_width = 800
pio.kaleido.scope.default_height = 600

SOLUTION_DIR = Path(__file__).parent / "pinn_solutions"

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
                if all(k in sol for k in required_keys):
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

# === Radar chart using actual concentration values as circumference labels ===
def plot_radar_concentration(solution, species="Cu", output_dir="figures"):
    Ly = solution['params']['Ly']
    times = solution['times']
    c_all = solution['c1_preds'] if species == "Cu" else solution['c2_preds']

    # Use the mean concentration profile along one direction (e.g., vertical center line)
    ny, nx = c_all.shape[2], c_all.shape[3]
    center_x = nx // 2
    c_axis = np.linspace(0, 1, ny)  # concentration position labels
    theta_labels = [f"{c_val:.2f}" for c_val in c_axis]

    fig = go.Figure()
    for t_idx, t_val in enumerate(times):
        c_profile = c_all[t_idx, :, center_x]
        fig.add_trace(go.Scatterpolar(
            r=[t_val] * len(c_profile),
            theta=theta_labels,
            mode="markers+text",
            text=[f"{c:.2f}" for c in c_profile],
            textposition="top center",
            name=f"t={t_val:.1f}s",
            hovertemplate="t=%{r:.2f}s<br>conc=%{text}<extra></extra>"
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

# === Polar chart (radius=time, circumference=concentration labels) ===
def plot_polar_concentration(solution, species="Cu", output_dir="figures"):
    Ly = solution['params']['Ly']
    times = solution['times']
    c_all = solution['c1_preds'] if species == "Cu" else solution['c2_preds']

    ny, nx = c_all.shape[2], c_all.shape[3]
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

# === Streamlit App ===
def main():
    st.title("🔬 PINN Visualization — Time vs Concentration (Cu/Ni)")

    solutions, lys = load_solutions(SOLUTION_DIR)
    if not solutions:
        st.error("No valid solutions found.")
        return

    # Choose Ly for single or comparison mode
    selected_lys = st.multiselect(
        "Select one or two Ly values to compare",
        options=lys,
        default=[lys[0]] if len(lys) == 1 else lys[:2],
        format_func=lambda x: f"{x:.1f} μm"
    )

    chart_type = st.selectbox("Chart Type", ["Radar Chart", "Polar Chart"])
    species = st.selectbox("Species", ["Cu", "Ni"])

    cols = st.columns(len(selected_lys)) if len(selected_lys) == 2 else [st]

    for idx, ly_choice in enumerate(selected_lys):
        with cols[idx]:
            st.markdown(f"### Ly = {ly_choice:.1f} μm")
            solution = next((sol for sol in solutions if abs(sol['params']['Ly'] - ly_choice) < 0.1), None)
            if not solution:
                st.warning(f"No solution found for Ly = {ly_choice:.1f}")
                continue

            if chart_type == "Radar Chart":
                fig, filename = plot_radar_concentration(solution, species)
            else:
                fig, filename = plot_polar_concentration(solution, species)

            st.plotly_chart(fig, use_container_width=True)

            html_path = os.path.join("figures", f"{filename}.html")
            if os.path.exists(html_path):
                st.download_button(
                    f"⬇️ Download {species} {chart_type} (HTML)",
                    data=open(html_path, "rb").read(),
                    file_name=f"{filename}.html",
                    mime="text/html"
                )
            png_path = os.path.join("figures", f"{filename}.png")
            if os.path.exists(png_path):
                st.download_button(
                    f"⬇️ Download {species} {chart_type} (PNG)",
                    data=open(png_path, "rb").read(),
                    file_name=f"{filename}.png",
                    mime="image/png"
                )
            else:
                st.info("PNG export unavailable (Chrome/Kaleido not installed).")

if __name__ == "__main__":
    main()
