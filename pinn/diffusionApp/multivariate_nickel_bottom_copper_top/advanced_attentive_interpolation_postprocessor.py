import os
import pickle
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

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
                if all(key in sol for key in required_keys):
                    if not (np.any(np.isnan(sol['c1_preds'])) or np.any(np.isnan(sol['c2_preds'])) or
                            np.all(sol['c1_preds'] == 0) or np.all(sol['c2_preds'] == 0)):
                        solutions.append(sol)
                        lys.append(sol['params']['Ly'])
            except Exception:
                continue
    return solutions, sorted(set(lys))

def plot_sunburst_chart(solution, time_index, output_dir="figures"):
    t_val = solution['times'][time_index]
    Ly = solution['params']['Ly']
    c1 = solution['c1_preds'][time_index]  # Cu concentration
    c2 = solution['c2_preds'][time_index]  # Ni concentration
    
    # Sample data at specific points
    points = ['Top', 'Bottom', 'Left', 'Right', 'Center']
    cu_values = [
        np.mean(c1[:, -1]),  # Top
        np.mean(c1[:, 0]),   # Bottom
        np.mean(c1[0, :]),   # Left
        np.mean(c1[-1, :]),  # Right
        c1[25, 25]           # Center
    ]
    ni_values = [
        np.mean(c2[:, -1]),  # Top
        np.mean(c2[:, 0]),   # Bottom
        np.mean(c2[0, :]),   # Left
        np.mean(c2[-1, :]),  # Right
        c2[25, 25]           # Center
    ]
    
    # Prepare data for sunburst
    labels = ['Solution'] + points + [f'Cu_{p}' for p in points] + [f'Ni_{p}' for p in points]
    parents = [''] + ['Solution'] * len(points) + [p for p in points for _ in range(2)]
    values = [0] + [0] * len(points) + cu_values + ni_values
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        textinfo="label+value",
        texttemplate="%{label}<br>%{value:.2e}",
        hovertemplate="%{label}: %{value:.2e} mol/cc<extra></extra>",
    ))
    
    fig.update_layout(
        title=f"Concentration Sunburst Chart<br>Ly = {Ly:.1f} μm, t = {t_val:.1f} s",
        margin=dict(t=50, l=0, r=0, b=0)
    )
    
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"sunburst_t_{t_val:.1f}_ly_{Ly:.1f}"
    fig.write_image(os.path.join(output_dir, f"{base_filename}.png"), format="png")
    fig.write_html(os.path.join(output_dir, f"{base_filename}.html"))
    
    return fig, base_filename

def plot_radar_chart(solution, time_indices, output_dir="figures"):
    Ly = solution['params']['Ly']
    categories = ['Top', 'Bottom', 'Left', 'Right', 'Center']
    
    fig = go.Figure()
    for t_idx in time_indices:
        t_val = solution['times'][t_idx]
        c1 = solution['c1_preds'][t_idx]
        c2 = solution['c2_preds'][t_idx]
        
        cu_values = [
            np.mean(c1[:, -1]),  # Top
            np.mean(c1[:, 0]),   # Bottom
            np.mean(c1[0, :]),   # Left
            np.mean(c1[-1, :]),  # Right
            c1[25, 25]           # Center
        ]
        ni_values = [
            np.mean(c2[:, -1]),  # Top
            np.mean(c2[:, 0]),   # Bottom
            np.mean(c2[0, :]),   # Left
            np.mean(c2[-1, :]),  # Right
            c2[25, 25]           # Center
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=cu_values,
            theta=categories,
            name=f'Cu, t={t_val:.1f}s',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatterpolar(
            r=ni_values,
            theta=categories,
            name=f'Ni, t={t_val:.1f}s',
            line=dict(color='red')
        ))
    
    fig.update_layout(
        title=f"Concentration Radar Chart<br>Ly = {Ly:.1f} μm",
        polar=dict(
            radialaxis=dict(
                visible=True,
                tickformat='.2e'
            )
        ),
        showlegend=True
    )
    
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"radar_ly_{Ly:.1f}"
    fig.write_image(os.path.join(output_dir, f"{base_filename}.png"), format="png")
    fig.write_html(os.path.join(output_dir, f"{base_filename}.html"))
    
    return fig, base_filename

def plot_polar_chart(solution, time_index, output_dir="figures"):
    t_val = solution['times'][time_index]
    Ly = solution['params']['Ly']
    c1 = solution['c1_preds'][time_index]
    c2 = solution['c2_preds'][time_index]
    
    # Sample along centerline (x = Lx/2)
    center_idx = 25
    theta = np.linspace(0, 2 * np.pi, len(c1[:, center_idx]))
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=c1[:, center_idx],
        theta=theta * 180 / np.pi,
        name='Cu',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatterpolar(
        r=c2[:, center_idx],
        theta=theta * 180 / np.pi,
        name='Ni',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title=f"Polar Chart of Centerline Concentrations<br>Ly = {Ly:.1f} μm, t = {t_val:.1f} s",
        polar=dict(
            radialaxis=dict(
                visible=True,
                tickformat='.2e'
            ),
            angularaxis=dict(
                rotation=90,
                direction="counterclockwise"
            )
        ),
        showlegend=True
    )
    
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"polar_t_{t_val:.1f}_ly_{Ly:.1f}"
    fig.write_image(os.path.join(output_dir, f"{base_filename}.png"), format="png")
    fig.write_html(os.path.join(output_dir, f"{base_filename}.html"))
    
    return fig, base_filename

def main():
    st.title("PINN Solution Visualization")

    # Load solutions
    solutions, lys = load_solutions(SOLUTION_DIR)

    if not solutions:
        st.error("No valid solution files found in pinn_solutions directory.")
        return

    st.subheader("Select Parameters")
    ly_choice = st.selectbox("Domain Height (Ly, μm)", options=lys, format_func=lambda x: f"{x:.1f}")

    # Select solution
    solution = None
    for sol in solutions:
        if abs(sol['params']['Ly'] - ly_choice) < 0.1:
            solution = sol
            break

    if not solution:
        st.error("No matching solution found.")
        return

    st.subheader("Chart Selection")
    chart_type = st.selectbox("Select Chart Type", options=['Sunburst Chart', 'Radar Chart', 'Polar Chart'])

    # Time selection
    time_indices = list(range(len(solution['times'])))
    if chart_type == 'Radar Chart':
        selected_times = st.multiselect(
            "Select Time Indices",
            options=time_indices,
            default=[0, len(time_indices)//4, len(time_indices)//2, 3*len(time_indices)//4, len(time_indices)-1],
            format_func=lambda x: f"t = {solution['times'][x]:.1f} s"
        )
    else:
        time_index = st.slider("Select Time Index", 0, len(solution['times'])-1, len(solution['times'])-1)

    # Generate and display chart
    if chart_type == 'Sunburst Chart':
        fig, filename = plot_sunburst_chart(solution, time_index)
        st.plotly_chart(fig, use_container_width=True)
        st.download_button(
            label="Download Sunburst Chart as HTML",
            data=open(os.path.join("figures", f"{filename}.html"), "rb").read(),
            file_name=f"{filename}.html",
            mime="text/html"
        )
        st.download_button(
            label="Download Sunburst Chart as PNG",
            data=open(os.path.join("figures", f"{filename}.png"), "rb").read(),
            file_name=f"{filename}.png",
            mime="image/png"
        )

    elif chart_type == 'Radar Chart' and selected_times:
        fig, filename = plot_radar_chart(solution, selected_times)
        st.plotly_chart(fig, use_container_width=True)
        st.download_button(
            label="Download Radar Chart as HTML",
            data=open(os.path.join("figures", f"{filename}.html"), "rb").read(),
            file_name=f"{filename}.html",
            mime="text/html"
        )
        st.download_button(
            label="Download Radar Chart as PNG",
            data=open(os.path.join("figures", f"{filename}.png"), "rb").read(),
            file_name=f"{filename}.png",
            mime="image/png"
        )

    elif chart_type == 'Polar Chart':
        fig, filename = plot_polar_chart(solution, time_index)
        st.plotly_chart(fig, use_container_width=True)
        st.download_button(
            label="Download Polar Chart as HTML",
            data=open(os.path.join("figures", f"{filename}.html"), "rb").read(),
            file_name=f"{filename}.html",
            mime="text/html"
        )
        st.download_button(
            label="Download Polar Chart as PNG",
            data=open(os.path.join("figures", f"{filename}.png"), "rb").read(),
            file_name=f"{filename}.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
