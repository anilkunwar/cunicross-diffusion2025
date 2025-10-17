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
            except Exception as e:
                st.warning(f"Failed to load {fname}: {str(e)}")
                continue
    if not solutions:
        st.error("No valid solutions found in pinn_solutions directory.")
    return solutions, sorted(set(lys))

def safe_save(fig, output_dir, base_filename):
    os.makedirs(output_dir, exist_ok=True)
    try:
        fig.write_image(os.path.join(output_dir, f"{base_filename}.png"), scale=2)
    except Exception as e:
        st.warning(f"PNG export failed: {e}")
    fig.write_html(os.path.join(output_dir, f"{base_filename}.html"))

# === Polar chart (centerline concentration as radial axis, multiple Ly) ===
def plot_polar_chart_centerline(solutions, selected_lys, species="Cu", step_size=1, output_dir="figures"):
    fig = go.Figure()
    colors = ['blue', 'red', 'green', 'purple', 'orange']  # Colors for different Ly values
    
    for idx, ly_choice in enumerate(selected_lys):
        solution = next((sol for sol in solutions if abs(sol['params']['Ly'] - ly_choice) < 0.1), None)
        if not solution:
            continue
            
        Ly = solution['params']['Ly']
        times = solution['times']
        c_all = solution['c1_preds'] if species == "Cu" else solution['c2_preds']
        center_idx = c_all.shape[1] // 2  # Lx/2 (assuming Lx=60 μm)
        
        # Extract centerline concentrations (x = Lx/2) with adjustable step size
        time_indices = range(0, len(times), step_size)
        theta = np.linspace(0, 2 * np.pi, len(time_indices))
        concentrations = [c_all[t_idx, center_idx, :] for t_idx in time_indices]
        
        # Normalize concentrations for visualization
        max_conc = max([np.max(c) for c in concentrations], default=1e-10)
        concentrations = [c / max_conc for c in concentrations]
        
        # Plot each time step
        for t_idx, conc in zip(time_indices, concentrations):
            fig.add_trace(go.Scatterpolar(
                r=conc,
                theta=theta * 180 / np.pi,
                mode="lines",
                name=f"Ly={Ly:.1f}, t={times[t_idx]:.1f}s",
                line=dict(color=colors[idx % len(colors)], width=1.5),
                hovertemplate=f"Ly={Ly:.1f}, t={times[t_idx]:.1f}s, y=%{{theta:.1f}}°, Conc=%{{r:.2e}} (norm)<extra></extra>",
                showlegend=(t_idx == time_indices[0])  # Show legend only for first time step per Ly
            ))
    
    # Customize layout
    fig.update_layout(
        title={
            'text': f"{species} Centerline Concentration Evolution (Lx/2 = 30 μm)<br>Normalized Concentrations",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        polar=dict(
            radialaxis=dict(
                visible=True,
                title="Normalized Concentration",
                range=[0, 1.1],
                tickformat=".2f"
            ),
            angularaxis=dict(
                visible=True,
                rotation=90,
                direction="counterclockwise",
                tickvals=[0, 90, 180, 270],
                ticktext=["0°", "90°", "180°", "270°"]
            )
        ),
        showlegend=True,
        font=dict(size=12),
        margin=dict(l=50, r=50, t=100, b=50),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    base_filename = f"polar_centerline_{species.lower()}_multi_ly"
    safe_save(fig, output_dir, base_filename)
    return fig, base_filename

# === Streamlit App ===
def main():
    st.title("🔬 PINN Centerline Concentration Evolution (Lx/2 = 30 μm)")

    # Load solutions
    solutions, lys = load_solutions(SOLUTION_DIR)
    if not solutions:
        return

    # Ly selection
    st.subheader("Select Domain Heights (Ly)")
    selected_lys = st.multiselect(
        "Select two or more Ly values to compare",
        options=lys,
        default=[lys[0]] if len(lys) == 1 else lys[:min(3, len(lys))],
        format_func=lambda x: f"{x:.1f} μm"
    )
    
    # Step size selection
    st.subheader("Time Step Size")
    step_size = st.slider("Select time step size", min_value=1, max_value=10, value=1, step=1)

    if len(selected_lys) < 2:
        st.warning("Please select at least two Ly values for comparison.")
        return

    # Generate and display polar charts for Cu and Ni
    st.subheader("Concentration Evolution")
    for species in ["Cu", "Ni"]:
        st.markdown(f"### {species} Concentration")
        fig, filename = plot_polar_chart_centerline(solutions, selected_lys, species, step_size)
        st.plotly_chart(fig, use_container_width=True)

        # Download buttons
        html_path = os.path.join("figures", f"{filename}.html")
        if os.path.exists(html_path):
            st.download_button(
                f"⬇️ Download {species} Polar Chart (HTML)",
                data=open(html_path, "rb").read(),
                file_name=f"{filename}.html",
                mime="text/html"
            )
        png_path = os.path.join("figures", f"{filename}.png")
        if os.path.exists(png_path):
            st.download_button(
                f"⬇️ Download {species} Polar Chart (PNG)",
                data=open(png_path, "rb").read(),
                file_name=f"{filename}.png",
                mime="image/png"
            )
        else:
            st.info("PNG export unavailable (Kaleido not installed).")

if __name__ == "__main__":
    main()
