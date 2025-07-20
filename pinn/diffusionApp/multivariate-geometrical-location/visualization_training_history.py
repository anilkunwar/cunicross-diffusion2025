import os
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import io
import matplotlib
from matplotlib import font_manager
import re

#SOLUTION_DIR = os.path.join(os.path.dirname(__file__), "pinn_solutions")
SOLUTION_DIR="pinn_solutions"

@st.cache_data
def load_solution_metadata(solution_dir):
    """Load metadata (Ly, C_Cu, C_Ni, filename) from all .pkl files."""
    metadata = []
    load_logs = []
    for fname in os.listdir(solution_dir):
        if fname.endswith(".pkl"):
            try:
                with open(os.path.join(solution_dir, fname), "rb") as f:
                    sol = pickle.load(f)
                if all(key in sol for key in ['params', 'loss_history']):
                    metadata.append({
                        'Ly': sol['params']['Ly'],
                        'C_Cu': sol['params']['C_Cu'],
                        'C_Ni': sol['params']['C_Ni'],
                        'filename': fname
                    })
                    load_logs.append(f"{fname}: Loaded successfully.")
                else:
                    load_logs.append(f"{fname}: Failed to load - missing required keys.")
            except Exception as e:
                load_logs.append(f"{fname}: Failed to load - {str(e)}.")
    return metadata, load_logs

def validate_filename_params(filename):
    """Extract Ly, C_Cu, C_Ni from filename."""
    pattern = r"solution_ly_([\d.]+)_ccu_([\d.e-]+)_cni_([\d.e-]+)_.*\.pkl"
    match = re.match(pattern, filename)
    if not match:
        return None, "Filename format invalid"
    ly_file, ccu_file, cni_file = map(float, match.groups())
    return {'Ly': ly_file, 'C_Cu': ccu_file, 'C_Ni': cni_file}, None

def plot_loss_history(loss_history, Ly, C_Cu, C_Ni, plot_params):
    """Generate a publication-quality loss plot with customizable parameters."""
    try:
        plt.style.use(plot_params['style'])
    except:
        plt.style.use('seaborn-v0_8')

    if plot_params['use_latex']:
        try:
            matplotlib.rc('text', usetex=True)
            matplotlib.rc('font', family='serif')
        except:
            st.warning("LaTeX rendering failed. Falling back to non-LaTeX mode.")
            matplotlib.rc('text', usetex=False)
            matplotlib.rc('font', family='sans-serif')
    else:
        matplotlib.rc('text', usetex=False)
        matplotlib.rc('font', family='sans-serif')

    fig, ax = plt.subplots(
        figsize=(plot_params['fig_width'], plot_params['fig_height']),
        dpi=plot_params['dpi']
    )

    epochs = loss_history['epochs']
    total_loss = loss_history['total']
    physics_loss = loss_history['physics']
    bottom_loss = loss_history['bottom']
    top_loss = loss_history['top']
    sides_loss = loss_history['sides']
    initial_loss = loss_history['initial']

    # Use the selected line style (solid or dotted) for all curves
    line_style = plot_params['line_style']

    ax.plot(epochs, total_loss, 
            label=plot_params['legend_labels']['total'], 
            color=plot_params['colors']['total'], 
            linewidth=plot_params['curve_thickness'], 
            linestyle=line_style)
    ax.plot(epochs, physics_loss, 
            label=plot_params['legend_labels']['physics'], 
            color=plot_params['colors']['physics'], 
            linewidth=plot_params['curve_thickness'], 
            linestyle=line_style)
    ax.plot(epochs, bottom_loss, 
            label=plot_params['legend_labels']['bottom'], 
            color=plot_params['colors']['bottom'], 
            linewidth=plot_params['curve_thickness'], 
            linestyle=line_style)
    ax.plot(epochs, top_loss, 
            label=plot_params['legend_labels']['top'], 
            color=plot_params['colors']['top'], 
            linewidth=plot_params['curve_thickness'], 
            linestyle=line_style)
    ax.plot(epochs, sides_loss, 
            label=plot_params['legend_labels']['sides'], 
            color=plot_params['colors']['sides'], 
            linewidth=plot_params['curve_thickness'], 
            linestyle=line_style)
    ax.plot(epochs, initial_loss, 
            label=plot_params['legend_labels']['initial'], 
            color=plot_params['colors']['initial'], 
            linewidth=plot_params['curve_thickness'], 
            linestyle=line_style)

    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=plot_params['label_font_size'], fontweight='bold')
    ax.set_ylabel('Loss', fontsize=plot_params['label_font_size'], fontweight='bold')

    if plot_params['use_latex']:
        title = (f'Training Loss for $L_y$ = {Ly:.1f} \\mu m, '
                 f'$C_{{Cu}}$ = {C_Cu:.1e}, $C_{{Ni}}$ = {C_Ni:.1e}')
    else:
        title = f'Training Loss for Ly = {Ly:.1f} μm, C_Cu = {C_Cu:.1e}, C_Ni = {C_Ni:.1e}'
    ax.set_title(title, fontsize=plot_params['title_font_size'], pad=15)

    if plot_params['show_grid']:
        ax.grid(True, which="both", ls="--", alpha=0.5)
    else:
        ax.grid(False)

    # Increase tick label size and thickness
    ax.tick_params(
        axis='both', 
        which='major', 
        labelsize=plot_params['tick_font_size'], 
        width=plot_params['tick_linewidth'],  # Control tick thickness
        length=6  # Slightly longer ticks for better visibility
    )

    for spine in ax.spines.values():
        spine.set_linewidth(plot_params['axes_thickness'])

    ax.legend(
        fontsize=plot_params['legend_font_size'],
        loc=plot_params['legend_position'],
        frameon=True,
        edgecolor='black',
        framealpha=1.0
    )

    plt.tight_layout()
    return fig

def export_loss_data(loss_history, Ly, C_Cu, C_Ni, legend_labels):
    """Export loss history as a CSV file with custom legend labels."""
    df = pd.DataFrame({
        'Epoch': loss_history['epochs'],
        legend_labels['total']: loss_history['total'],
        legend_labels['physics']: loss_history['physics'],
        legend_labels['bottom']: loss_history['bottom'],
        legend_labels['top']: loss_history['top'],
        legend_labels['sides']: loss_history['sides'],
        legend_labels['initial']: loss_history['initial']
    })
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode('utf-8')
    filename = f"loss_data_ly_{Ly:.1f}_ccu_{C_Cu:.1e}_cni_{C_Ni:.1e}.csv"
    return csv_bytes, filename

def main():
    st.title("Customizable Loss History Visualization for Cross-Diffusion Model")

    # Clear cache button
    if st.sidebar.button("Clear Metadata Cache"):
        load_solution_metadata.clear()
        st.sidebar.success("Cache cleared. Reload the app to refresh metadata.")

    # Sidebar for customization
    st.sidebar.header("Plot Customization")
    
    plot_style = st.sidebar.selectbox("Plot Style", ['seaborn-v0_8', 'ggplot', 'bmh', 'classic'], index=0)
    title_font_size = st.sidebar.slider("Title Font Size", min_value=10, max_value=44, value=16, step=1)
    label_font_size = st.sidebar.slider("Axis Label Font Size", min_value=8, max_value=40, value=20, step=1)
    tick_font_size = st.sidebar.slider("Tick Label Font Size", min_value=6, max_value=36, value=16, step=1)
    tick_linewidth = st.sidebar.slider("Tick Line Thickness", min_value=0.5, max_value=10.0, value=3.0, step=0.1)
    legend_font_size = st.sidebar.slider("Legend Font Size", min_value=6, max_value=36, value=13, step=1)
    axes_thickness = st.sidebar.slider("Axes Line Thickness", min_value=0.5, max_value=10.0, value=3.0, step=0.1)
    curve_thickness = st.sidebar.slider("Curve Line Thickness", min_value=0.5, max_value=10.0, value=3.50, step=0.1)
    line_style = st.sidebar.selectbox("Line Style", ['Solid', 'Dotted'], format_func=lambda x: x, index=0)
    line_style = '-' if line_style == 'Solid' else ':'

    st.sidebar.subheader("Curve Colors")
    color_total = st.sidebar.color_picker("Total Loss Color", value='#1f77b4')
    color_physics = st.sidebar.color_picker("Physics Loss Color", value='#ff7f0e')
    color_bottom = st.sidebar.color_picker("Bottom Boundary Loss Color", value='#2ca02c')
    color_top = st.sidebar.color_picker("Top Boundary Loss Color", value='#d62728')
    color_sides = st.sidebar.color_picker("Side Boundaries Loss Color", value='#9467bd')
    color_initial = st.sidebar.color_picker("Initial Condition Loss Color", value='#8c564b')

    legend_position = st.sidebar.selectbox("Legend Position", [
        'upper right', 'upper left', 'lower right', 'lower left',
        'center', 'center left', 'center right', 'upper center', 'lower center'
    ], index=0)

    st.sidebar.subheader("Legend Labels")
    legend_label_total = st.sidebar.text_input("Total Loss Label", value="Total Loss")
    legend_label_physics = st.sidebar.text_input("Physics Loss Label", value="PDE Residual")
    legend_label_bottom = st.sidebar.text_input("Bottom Boundary Loss Label", value="Bottom BC Loss")
    legend_label_top = st.sidebar.text_input("Top Boundary Loss Label", value="Top BC Loss")
    legend_label_sides = st.sidebar.text_input("Side Boundaries Loss Label", value="Side BC Loss")
    legend_label_initial = st.sidebar.text_input("Initial Condition Loss Label", value="Initial Condition Loss")

    show_grid = st.sidebar.checkbox("Show Grid", value=True)
    use_latex = st.sidebar.checkbox("Use LaTeX for Text Rendering", value=False)
    fig_width = st.sidebar.slider("Figure Width (inches)", min_value=4.0, max_value=12.0, value=8.0, step=0.5)
    fig_height = st.sidebar.slider("Figure Height (inches)", min_value=3.0, max_value=9.0, value=6.0, step=0.5)
    dpi = st.sidebar.slider("Export DPI", min_value=100, max_value=600, value=300, step=50)

    plot_params = {
        'style': plot_style,
        'title_font_size': title_font_size,
        'label_font_size': label_font_size,
        'tick_font_size': tick_font_size,
        'tick_linewidth': tick_linewidth,
        'legend_font_size': legend_font_size,
        'axes_thickness': axes_thickness,
        'curve_thickness': curve_thickness,
        'line_style': line_style,
        'colors': {
            'total': color_total,
            'physics': color_physics,
            'bottom': color_bottom,
            'top': color_top,
            'sides': color_sides,
            'initial': color_initial
        },
        'legend_position': legend_position,
        'legend_labels': {
            'total': legend_label_total,
            'physics': legend_label_physics,
            'bottom': legend_label_bottom,
            'top': legend_label_top,
            'sides': legend_label_sides,
            'initial': legend_label_initial
        },
        'show_grid': show_grid,
        'use_latex': use_latex,
        'fig_width': fig_width,
        'fig_height': fig_height,
        'dpi': dpi
    }

    metadata, load_logs = load_solution_metadata(SOLUTION_DIR)

    if load_logs:
        st.subheader("Solution Load Log")
        selected_log = st.selectbox("View load status for solutions (Non-functional List: Refer to the Select Parameters section)", load_logs, index=0)
        st.write(selected_log)
    else:
        st.warning("No solution files found in pinn_solutions directory.")

    if not metadata:
        st.error("No valid solution files found in pinn_solutions directory.")
        return

    lys = sorted(set(m['Ly'] for m in metadata))
    c_cus = sorted(set(m['C_Cu'] for m in metadata))
    c_nis = sorted(set(m['C_Ni'] for m in metadata))

    st.subheader("Select Parameters")
    ly_choice = st.selectbox("Domain Height (Ly, μm)", options=lys, format_func=lambda x: f"{x:.1f}")
    c_cu_choice = st.selectbox("Cu Boundary Concentration (mol/cc)", options=c_cus, format_func=lambda x: f"{x:.1e}")
    c_ni_choice = st.selectbox("Ni Boundary Concentration (mol/cc)", options=c_nis, format_func=lambda x: f"{x:.1e}")

    st.write(f"Selected Parameters: Ly={ly_choice:.1f}, C_Cu={c_cu_choice:.1e}, C_Ni={c_ni_choice:.1e}")

    matching_solutions = [
        m for m in metadata
        if abs(m['Ly'] - ly_choice) < 1e-6 and abs(m['C_Cu'] - c_cu_choice) < 1e-6 and abs(m['C_Ni'] - c_ni_choice) < 1e-6
    ]

    if not matching_solutions:
        st.error(f"No solution found for Ly={ly_choice:.1f}, C_Cu={c_cu_choice:.1e}, C_Ni={c_ni_choice:.1e}.")
        return

    solution_metadata = matching_solutions[0]
    solution_filename = os.path.join(SOLUTION_DIR, solution_metadata['filename'])

    try:
        with open(solution_filename, 'rb') as f:
            solution = pickle.load(f)
        loss_history = solution['loss_history']

        # Validate loaded parameters
        loaded_params = solution['params']
        filename_params, filename_error = validate_filename_params(solution_metadata['filename'])
        if filename_error:
            st.error(f"Invalid filename format: {solution_metadata['filename']}")
            return

        st.write(f"Loaded file: {solution_metadata['filename']}")
        st.write("Parameter Validation:")
        st.write({
            'Selected Parameters': f"Ly={ly_choice:.1f}, C_Cu={c_cu_choice:.1e}, C_Ni={c_ni_choice:.1e}",
            'File Parameters (from filename)': f"Ly={filename_params['Ly']:.1f}, C_Cu={filename_params['C_Cu']:.1e}, C_Ni={filename_params['C_Ni']:.1e}",
            'Loaded Parameters (from file)': f"Ly={loaded_params['Ly']:.1f}, C_Cu={loaded_params['C_Cu']:.1e}, C_Ni={loaded_params['C_Ni']:.1e}"
        })

        # Check for parameter mismatches
        if (abs(ly_choice - loaded_params['Ly']) > 1e-6 or
            abs(c_cu_choice - loaded_params['C_Cu']) > 1e-6 or
            abs(c_ni_choice - loaded_params['C_Ni']) > 1e-6):
            st.warning("Mismatch between selected parameters and loaded file parameters. The plot may not reflect the selected parameters.")

        if (abs(filename_params['Ly'] - loaded_params['Ly']) > 1e-6 or
            abs(filename_params['C_Cu'] - loaded_params['C_Cu']) > 1e-6 or
            abs(filename_params['C_Ni'] - loaded_params['C_Ni']) > 1e-6):
            st.warning("Mismatch between filename parameters and loaded file parameters.")

        st.write("Loss History Summary:")
        st.write({
            'Number of Epochs': len(loss_history['epochs']),
            'Final Total Loss': loss_history['total'][-1],
            'Final Physics Loss': loss_history['physics'][-1],
            'Final Bottom BC Loss': loss_history['bottom'][-1],
            'Final Top BC Loss': loss_history['top'][-1],
            'Final Side BC Loss': loss_history['sides'][-1],
            'Final Initial Condition Loss': loss_history['initial'][-1]
        })
    except Exception as e:
        st.error(f"Failed to load solution {solution_metadata['filename']}: {str(e)}")
        return

    st.subheader("Loss History Plot")
    try:
        fig = plot_loss_history(loss_history, ly_choice, c_cu_choice, c_ni_choice, plot_params)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Failed to generate plot: {str(e)}")
        return

    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=plot_params['dpi'], bbox_inches='tight')
    img_buffer.seek(0)
    st.download_button(
        label="Download Loss Plot as PNG",
        data=img_buffer,
        file_name=f"loss_plot_ly_{ly_choice:.1f}_ccu_{c_cu_choice:.1e}_cni_{c_ni_choice:.1e}.png",
        mime="image/png"
    )

    csv_bytes, csv_filename = export_loss_data(loss_history, ly_choice, c_cu_choice, c_ni_choice, plot_params['legend_labels'])
    st.download_button(
        label="Download Loss Data as CSV",
        data=csv_bytes,
        file_name=csv_filename,
        mime="text/csv"
    )

    plt.close(fig)

if __name__ == "__main__":
    main()
