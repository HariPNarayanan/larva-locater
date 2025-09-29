import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from IPython.display import HTML

def animate_trajectories(
    df, trail=20, interval=50, cmap='tab10', save_path=None, fps=20
):
    """
    Animate (X,Y) trajectories of multiple individuals with fading trails,
    a live counter of individuals, and metadata in the title.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ['Frame', 'Individual', 'X', 'Y'].
        Can also contain ['Genotype', 'Odour', 'Concentration'] for titles.
    trail : int, optional
        Number of past frames to keep visible as a fading trail.
    interval : int, optional
        Delay between frames in milliseconds (controls playback speed).
    cmap : str, optional
        Matplotlib colormap for assigning distinct colors per individual.
    save_path : str or None, optional
        If provided, saves the animation to this file (e.g. 'out.mp4').
    fps : int, optional
        Frames per second for saving video.
    """

    # Ensure proper sorting
    df = df.sort_values(['Frame', 'Individual']).copy()
    frames = df['Frame'].unique()
    ids = df['Individual'].unique()

    # Extract metadata (if present)
    genotype = df['Genotype'].iloc[0] if 'Genotype' in df.columns else None
    odour = df['Odour'].iloc[0] if 'Odour' in df.columns else None
    concentration = df['Concentration'].iloc[0] if 'Concentration' in df.columns else None

    # Build base title
    meta_title = ' | '.join(
        str(x) for x in [genotype, odour, concentration] if x is not None
    )

    # Assign each individual a unique color
    colors = cm.get_cmap(cmap, len(ids))
    id_color = {i: colors(j) for j, i in enumerate(ids)}

    # Figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(df['X'].min() - 10, df['X'].max() + 10)
    ax.set_ylim(df['Y'].min() - 10, df['Y'].max() + 10)
    ax.set_aspect('equal', 'box')

    def update(frame_idx):
        ax.clear()
        ax.set_xlim(df['X'].min() - 10, df['X'].max() + 10)
        ax.set_ylim(df['Y'].min() - 10, df['Y'].max() + 10)
        ax.set_aspect('equal', 'box')

        current_frame = frames[frame_idx]

        # Count number of individuals in this frame
        present_ids = df.loc[df['Frame'] == current_frame, 'Individual'].nunique()

        # Title with metadata + current frame
        title_parts = []
        if meta_title:
            title_parts.append(meta_title)
        title_parts.append(f'Frame {current_frame}')
        ax.set_title(' | '.join(title_parts))

        # Counter text box
        ax.text(
            0.02, 0.95,
            f'Individuals present: {present_ids}',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

        # Draw individuals with trails
        for i in ids:
            hist = df[
                (df['Individual'] == i) &
                (df['Frame'] <= current_frame) &
                (df['Frame'] >= current_frame - trail)
            ]

            if hist.empty:
                continue

            # Age factor: 0=newest, 1=oldest
            ages = (current_frame - hist['Frame']) / trail

            for (x, y, a) in zip(hist['X'], hist['Y'], ages):
                ax.plot(
                    x, y, 'o',
                    color=id_color[i],
                    alpha=1 - a * 0.9,
                    markersize=6 if a == 0 else 4,
                )

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=interval, blit=False
    )

    # Save to file if requested
    if save_path:
        ani.save(save_path, fps=fps, dpi=150)

    plt.close(fig)  # prevent duplicate static plot in Jupyter
    return HTML(ani.to_jshtml())
