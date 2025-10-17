import os
import pickle
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import plotly.io as pio
import logging

# === Logging setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Kaleido defaults (for export safety) ===
pio.kaleido.scope.default_format = "png"
pio.kaleido.scope.default_width = 1000
pio.kaleido.scope.default_height = 800

# === Paths and boundary constants ===
SOLUTION_DIR = Path(__file__).parent / "pinn_solutions"
C_CU_TOP = 0.0
C_CU_BOTTOM = 1.6e-3
C_NI_TOP = 1.25e-3
C_NI_BOTTOM = 0.0


# === Load solutions ===
@st.cache_data
def load_solutions(solution_dir):
    solutions, lys, logs = [], [], []
    for fname in os.listdir(solution_dir):
        if fname.endswith(".pkl"):
            try:
                with open(solution_dir / fname, "rb") as f:
                    sol = pickle.load(f)
                required = ["params", "X", "Y", "c1_preds", "c2_preds", "times"]
                if not all(k in sol for k in required):
                    logs.append(f"{fname}: missing keys → skipped.")
                    continue
                if np.any(np.isnan(sol["c1_preds"])) or np.any(np.isnan(sol["c2_preds"])):
                    logs.append(f"{fname}: contains NaNs → skipped.")
                    continue
                solutions.append(sol)
                lys.append(sol["params"]["Ly"])
                logs.append(f"{fname}: loaded successfully.")
            except Exception as e:
                logs.append(f"{fname}: failed ({e})")
                logger.error(f"Load error in {fname}: {e}")
    return solutions, sorted(set(lys)), logs


def safe_save(fig, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f"{name}.html")
    fig.write_html(html_path)
    try:
        fig.write_image(os.path.join(output_dir, f"{name}.png"))
    except Exception as e:
        st.warning(f"⚠️ PNG export failed: {e}")
    return html_path


# === Enforce boundary conditions ===
def enforce_boundary_conditions(solution):
    for t in range(len(solution["times"])):
        c1 = np.array(solution["c1_preds"][t])
        c2 = np.array(solution["c2_preds"][t])

        # Ensure (Ny, Nx)
        if c1.shape[0] < c1.shape[1]:
            c1, c2 = c1.T, c2.T

        # Apply BCs
        c1[-1, :] = C_CU_TOP
        c1[0, :] = C_CU_BOTTOM
        c2[-1, :] = C_NI_TOP
        c2[0, :] = C_NI_BOTTOM
        c1[:, 0] = c1[:, 1]
        c1[:, -1] = c1[:, -2]
        c2[:, 0] = c2[:, 1]
        c2[:, -1] = c2[:, -2]

        solution["c1_preds"][t] = c1
        solution["c2_preds"][t] = c2

    # t=0 initialization
    solution["c1_preds"][0] = np.zeros_like(solution["c1_preds"][0])
    solution["c2_preds"][0] = np.zeros_like(solution["c2_preds"][0])
    return solution


# === Utility: safe centerline extraction ===
def get_centerline_conc(solution, species="Cu"):
    c_all = np.array(solution["c1_preds"] if species == "Cu" else solution["c2_preds"])

    # Handle (Nt, Ny, Nx) or (Nt, Nx, Ny)
    if c_all.ndim == 4:
        c_all = c_all[0]
    if c_all.shape[1] > c_all.shape[2]:  # assume (Nt, Ny, Nx)
        ny, nx = c_all.shape[1], c_all.shape[2]
        center_idx_x = nx // 2
        data = c_all[:, :, center_idx_x]  # (Nt, Ny)
    else:  # assume (Nt, Nx, Ny)
        nx, ny = c_all.shape[1], c_all.shape[2]
        center_idx_x = nx // 2
        data = c_all[:, center_idx_x, :]  # (Nt, Ny)
    return data


# === Radar chart ===
def plot_radar_chart(solutions, selected_lys, species="Cu", output_dir="figures"):
    categories = [f"y={lab}" for lab in ["0", "Ly/4", "Ly/2", "3Ly/4", "Ly"]]
    fig = go.Figure()
    colors = ["blue", "red"]

    for idx, ly in enumerate(selected_lys):
        sol = next((s for s in solutions if abs(s["params"]["Ly"] - ly) < 1e-6), None)
        if sol is None:
            continue
        conc_all = get_centerline_conc(sol, species)
        ny = conc_all.shape[1]
        y_ids = [0, ny // 4, ny // 2, 3 * ny // 4, ny - 1]
        times = sol["times"]
        time_indices = np.linspace(0, len(times) - 1, 5, dtype=int)

        for t_idx in time_indices:
            t = times[t_idx]
            cvals = [conc_all[t_idx, y] for y in y_ids]
            fig.add_trace(go.Scatterpolar(
                r=[t] * len(categories),
                theta=categories,
                mode="markers+lines+text",
                text=[f"{v:.2e}" for v in cvals],
                textposition="top center",
                marker=dict(size=8, color=cvals,
                            colorscale="Viridis" if species == "Cu" else "Magma",
                            cmin=0, cmax=C_CU_BOTTOM if species == "Cu" else C_NI_TOP,
                            colorbar_title=f"{species} conc."),
                line=dict(color=colors[idx], width=2),
                name=f"{species}, Ly={ly:.0f}, t={t:.1f}s",
                hovertemplate=f"{species}<br>Ly={ly:.0f}μm<br>t={t:.1f}s<br>%{{theta}} → %{{text}}<extra></extra>"
            ))

    fig.update_layout(
        title=f"{species} Concentration Radar Chart<br>Time (radial) — Position (angular)",
        polar=dict(radialaxis=dict(visible=True, title="Time (s)")),
        showlegend=True,
        margin=dict(l=40, r=40, t=100, b=40)
    )

    fname = f"radar_{species.lower()}_ly_" + "_".join([str(ly) for ly in selected_lys])
    safe_save(fig, output_dir, fname)
    return fig, fname


# === Streamlit main ===
def main():
    st.title("🔬 Cu & Ni Concentration Radar Charts")

    sols, lys, logs = load_solutions(SOLUTION_DIR)
    if not sols:
        st.error("No valid solutions found.")
        return

    with st.expander("📜 Load log"):
        for log in logs:
            st.write(log)

    selected_lys = st.multiselect("Select Ly values to compare", lys, default=lys[:2])
    sols = [enforce_boundary_conditions(s) for s in sols]

    # Cu chart
    st.subheader("🧿 Cu Concentration Radar Chart")
    fig_cu, name_cu = plot_radar_chart(sols, selected_lys, "Cu")
    st.plotly_chart(fig_cu, use_container_width=True)

    html_cu = f"figures/{name_cu}.html"
    if os.path.exists(html_cu):
        st.download_button("⬇ Download Cu Chart (HTML)",
                           data=open(html_cu, "rb").read(),
                           file_name=os.path.basename(html_cu),
                           mime="text/html")

    # Ni chart
    st.subheader("🧲 Ni Concentration Radar Chart")
    fig_ni, name_ni = plot_radar_chart(sols, selected_lys, "Ni")
    st.plotly_chart(fig_ni, use_container_width=True)

    html_ni = f"figures/{name_ni}.html"
    if os.path.exists(html_ni):
        st.download_button("⬇ Download Ni Chart (HTML)",
                           data=open(html_ni, "rb").read(),
                           file_name=os.path.basename(html_ni),
                           mime="text/html")


if __name__ == "__main__":
    main()
