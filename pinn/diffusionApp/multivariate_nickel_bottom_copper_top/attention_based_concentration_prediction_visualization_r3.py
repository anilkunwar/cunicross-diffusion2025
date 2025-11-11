def plot_sunburst(data, title, cmap, vmin, vmax, log_scale, ly_dir, fname):
    fig, ax = plt.subplots(figsize=(9,9), subplot_kw=dict(projection='polar'))

    # --- 9 theta edges for 8 spokes ---
    theta_edges = np.linspace(0, 2*np.pi, len(LY_SPOKES) + 1)
    r_edges     = np.linspace(0, 1, N_TIME + 1)

    # --- meshgrid: (51, 9) ---
    Theta, R = np.meshgrid(theta_edges, r_edges)

    # --- reverse time direction if needed ---
    if ly_dir == "top→bottom":
        R = R[::-1]           # flip rows
        data = data[::-1, :]  # flip data rows

    # --- color norm ---
    norm = LogNorm(vmin=max(vmin, 1e-9), vmax=vmax) if log_scale else Normalize(vmin=vmin, vmax=vmax)

    # --- pcolormesh: C is (50, 8), mesh is (51, 9) → perfect ---
    im = ax.pcolormesh(Theta, R, data, cmap=cmap, norm=norm, shading='auto')

    # --- spoke labels (at center of each sector) ---
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    ax.set_xticks(theta_centers)
    ax.set_xticklabels([f"{ly}" for ly in LY_SPOKES], fontsize=13, fontweight='bold')

    # --- time labels ---
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0', '50', '100', '150', '200'], fontsize=11)
    ax.set_ylim(0, 1)

    # --- style ---
    ax.grid(True, color='w', linewidth=1.2, alpha=0.7)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=25)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label('Concentration (mol/cc)', fontsize=13)

    plt.tight_layout()
    png = os.path.join(FIGURE_DIR, f"{fname}.png")
    pdf = os.path.join(FIGURE_DIR, f"{fname}.pdf")
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(pdf, bbox_inches='tight')
    plt.close()
    return fig, png, pdf
