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

# === Utility: compute mean values ===
def extract_values(c):
    """Return dict of mean concentrations at key regions."""
    return {
        'Top': np.mean(c[:, -1]),
        'Bottom': np.mean(c[:, 0]),
        'Left': np.mean(c[0, :]),
        'Right': np.mean(c[-1, :]),
        'Center': c[c.shape[0] // 2, c.shape[1] // 2]
    }

# === Radar chart (separate Cu/Ni, time as radius) ===
def plot_radar_chart_species(solution, species="Cu", output_dir="figures"):
    Ly = solution['params']['Ly']
    times = solution['times']
    c_all = solution['c1_preds'] if species == "Cu" else solution['c2_preds']
    categories = ['Top', 'Bottom', 'Left', 'Right', 'Center']
    
    fig = go.Figure()
    # Concentrations as categories, time as radial values
    for t_idx, t_val in enumerate(times):
        vals = extract_values(c_all[t_idx])
        r = [t_val] * len(categories)  # Time as radius
        fig.add_trace(go.Scatterpolar(
            r=r,
            theta=categories,
            mode="markers",
            marker=dict(
                size=10,
                color=list(vals.values()),
                colorscale="Viridis",
                showscale=True,
                colorbar_title=f"{species} Conc."
            ),
            name=f"t={t_val:.1f}s",
            text=[f"{v:.2e}" for v in vals.values()],
            textposition="top center",
            hovertemplate=f"t={t_val:.1f}s<br>%{{theta}}: %{{marker.color:.2e}}<extra></extra>"
        ))
    
    fig.update_layout(
        title=f"{species} Concentration Radar Chart<br>Ly={Ly:.1f} μm (Time → Radius)",
        polar=dict(
            radialaxis=dict(
                visible=True,
                title="Time (s)"
            )
        ),
        showlegend=True
    )
    base_filename = f"radar_{species.lower()}_ly_{Ly:.1f}"
    safe_save(fig, output_dir, base_filename)
    return fig, base_filename

# === Polar chart (separate Cu/Ni, time as radius) ===
def plot_polar_chart_species(solution, species="Cu", output_dir="figures"):
    Ly = solution['params']['Ly']
    times = solution['times']
    c_all = solution['c1_preds'] if species == "Cu" else solution['c2_preds']
    center_idx = c_all.shape[2] // 2
    theta = np.linspace(0, 2 * np.pi, c_all.shape[1])
    
    fig = go.Figure()
    # Concentrations along centerline, time as radius
    for t_idx, t_val in enumerate(times):
        z = c_all[t_idx][:, center_idx]
        r = np.full_like(theta, t_val)  # Time as radius
        fig.add_trace(go.Scatterpolar(
            r=r,
            theta=theta * 180 / np.pi,
            mode="markers",
            marker=dict(
                size=6,
                color=z,
                colorscale="Viridis",
                showscale=True,
                colorbar_title=f"{species} Conc."
            ),
            name=f"t={t_val:.1f}s",
            hovertemplate=f"θ=%{{theta:.1f}}°, c=%{{marker.color:.2e}}, t={t_val:.1f}s<extra></extra>"
        ))
    
    fig.update_layout(
        title=f"{species} Polar Chart (Time → Radius)<br>Ly={Ly:.1f} μm",
        polar=dict(
            radialaxis=dict(
                visible=True,
                title="Time (s)"
            ),
            angularaxis=dict(
                rotation=90,
                direction="counterclockwise"
            )
        ),
        showlegend=False
    )
    base_filename = f"polar_{species.lower()}_ly_{Ly:.1f}"
    safe_save(fig, output_dir, base_filename)
    return fig, base_filename

# === Sunburst (separate species, time as value weight) ===
def plot_sunburst_chart_species(solution, species="Cu", output_dir="figures"):
    t_val = solution['times'][-1]
    Ly = solution['params']['Ly']
    c = solution['c1_preds'][-1] if species == "Cu" else solution['c2_preds'][-1]
    vals = extract_values(c)
    
    labels = ['Solution'] + list(vals.keys())
    parents = [''] + ['Solution'] * len(vals)
    values = [t_val] + [t_val * v for v in vals.values()]  # Time-weighted concentrations
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        textinfo="label+value",
        texttemplate="%{label}<br>%{value:.2e}",
        hovertemplate="%{label}: %{value:.2e} (t-weighted)<extra></extra>"
    ))
    fig.update_layout(
        title=f"{species} Sunburst Chart (Time-weighted)<br>Ly={Ly:.1f} μm, t={t_val:.1f}s"
    )
    base_filename = f"sunburst_{species.lower()}_ly_{Ly:.1f}"
    safe_save(fig, output_dir, base_filename)
    return fig, base_filename

# === Streamlit App ===
def main():
    st.title("🔬 PINN Solution Visualization — Time as Radial Measure")
    solutions, lys = load_solutions(SOLUTION_DIR)
    if not solutions:
        st.error("No valid solutions found.")
        return

    # Ly selection (comparison mode)
    st.subheader("Select Domain Heights (Ly)")
    selected_lys = st.multiselect(
        "Select one or two Ly values to compare",
        options=lys,
        default=[lys[0]] if len(lys) == 1 else lys[:2],
        format_func=lambda x: f"{x:.1f} μm"
    )

    st.subheader("Chart Type & Species")
    chart_type = st.selectbox("Chart Type", ["Radar Chart", "Polar Chart", "Sunburst Chart"])
    species = st.selectbox("Species", ["Cu", "Ni"])

    for ly_choice in selected_lys:
        st.markdown(f"### Results for Ly = {ly_choice:.1f} μm")
        solution = next((sol for sol in solutions if abs(sol['params']['Ly'] - ly_choice) < 0.1), None)
        if not solution:
            st.warning(f"No solution found for Ly = {ly_choice:.1f}")
            continue

        if chart_type == "Radar Chart":
            fig, filename = plot_radar_chart_species(solution, species)
        elif chart_type == "Polar Chart":
            fig, filename = plot_polar_chart_species(solution, species)
        else:
            fig, filename = plot_sunburst_chart_species(solution, species)
        
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
