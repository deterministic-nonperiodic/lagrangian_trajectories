import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as lines

from sklearn.mixture import GaussianMixture
from tools import compute_intersections

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection

# Apply Nature-style theme using seaborn and matplotlib after kernel reset
sns.set_theme(style="whitegrid", font_scale=1.1)

# Update Matplotlib rcParams for Nature-quality styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.6,
    'xtick.major.size': 3,
    'xtick.major.width': 0.6,
    'ytick.major.size': 3,
    'ytick.major.width': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': True,
    'legend.framealpha': 0.85,
    'legend.edgecolor': 'black',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.bbox': 'tight',
    'savefig.dpi': 300,
})


def plot_faded_trajectory(ax, lon, lat, times, color='gray', linewidth=0.8,
                          transform=None, alpha_start=0.3, alpha_end=0.01):
    """
    Plot a trajectory with fading transparency using LineCollection.
    """
    from matplotlib.colors import to_rgba

    # Create segments
    points = np.array([lon, lat]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Alpha fades with time
    n = len(segments)
    alphas = np.linspace(alpha_start, alpha_end, n)

    rgba = np.tile(to_rgba(color), (n, 1))
    rgba[:, 3] = alphas

    lc = LineCollection(segments, colors=rgba, linewidths=linewidth, transform=transform)
    ax.add_collection(lc)


def custom_viridis():
    # Get the viridis colormap
    viridis = plt.cm.get_cmap('viridis', 256)

    # Convert to the array of colors
    viridis_colors = viridis(np.linspace(0, 1, 256))

    # Create a new colormap that fades in from white
    white_to_start = 100
    white_to_viridis = np.vstack([
        np.linspace([1, 1, 1, 1], viridis_colors[white_to_start], white_to_start),
        viridis_colors[white_to_start:]  # rest of viridis
    ])

    return mcolors.LinearSegmentedColormap.from_list("custom_viridis", white_to_viridis)


def generate_representative_particles(trajectories, orbit_df):
    """
    For each particle, select the ensemble member that intersects the orbit, or use ensemble 0.
    Returns:
        A dictionary mapping particle index to the selected ensemble index.
    """
    intersect_mask, intersect_indices = compute_intersections(trajectories, orbit_df,
                                                              horiz_tol_km=50, vert_tol_km=5)
    selected = {}
    for p in trajectories.particle.values:
        # Filter for current particle
        mask = intersect_indices[:, 1] == p
        intersect_ens = intersect_indices[mask][:, 2]
        intersect_time = intersect_indices[mask][:, 0]

        if intersect_ens.size > 0:
            selected[p] = (intersect_ens[0].item(), intersect_time[0].item())
        else:
            selected[p] = (0, trajectories.time.size - 1)  # fallback
    selected = dict(sorted(selected.items()))
    return selected


def visualize_trajectories_percentile_kde(trajectories, wind=None, particle_subset=None,
                                          orbit=None, figure_name=None):
    # Visualizes ensemble back-trajectories with wind streamlines and KDE of endpoints.
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True,
                           subplot_kw={"projection": ccrs.PlateCarree()})

    # Map settings
    map_extent = [-30.01, 20.01, 41., 69.]  # [lon_min, lon_max, lat_min, lat_max]
    ax.set_extent(map_extent)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    ax.add_feature(cfeature.LAND, color='gray', alpha=0.25)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3)

    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {'size': 9, 'color': 'gray'}

    # Wind streamlines (if provided)
    altitude_km = trajectories.isel(particle=0).z.isel(time=0).mean('ensemble').item()
    if wind is not None:
        wind_avg = (
            wind.sel(time=slice(trajectories.time[-1], trajectories.time[0]))
            .sel(z_mc=1e3 * altitude_km, method='nearest')
            .mean(dim='time')
        )
        wind_avg.plot.streamplot(x="lon", y="lat", u="u", v="v", ax=ax,
                                 transform=ccrs.PlateCarree(), density=1.,
                                 color='gray', linewidth=0.1)

    # Colormap for altitude
    cmap_altitude = sns.color_palette("Spectral_r", as_cmap=True)
    norm_altitude = mcolors.Normalize(vmin=96, vmax=104)

    # Orbit plot (if provided)
    if orbit is not None:
        ax.plot(orbit['GLon'], orbit['GLat'], color='black', linewidth=1.6,
                transform=ccrs.PlateCarree())

    # Particle subset
    particles = particle_subset or trajectories.particle.values

    if orbit is not None:
        representative_ensemble = generate_representative_particles(trajectories, orbit)
    else:
        representative_ensemble = {p: (0, trajectories.time.size - 1) for p in particles}

    # Plot each particle's trajectory
    particle_patches = []
    markers = ['o', 's', 'd', 'o', 'v', '<', '>', 'p', 'h', 'D', 'P', 'X']

    for i in particles:
        particle = trajectories.sel(particle=i)
        representative = particle.isel(ensemble=representative_ensemble[i][0],
                                       time=slice(None, representative_ensemble[i][1] + 1))

        # Full and representative trajectories
        ax.plot(particle.lon, particle.lat, color='gray', alpha=0.01,
                linewidth=0.8, zorder=1, transform=ccrs.PlateCarree())
        ax.plot(representative.lon, representative.lat, color='black', alpha=0.6,
                linewidth=1.0, zorder=3, transform=ccrs.PlateCarree())

        # Sampled altitude markers
        sampled = representative.isel(time=slice(None, None, 3))
        sampled.plot.scatter(x="lon", y="lat", hue='z', cmap=cmap_altitude,
                             norm=norm_altitude, marker=markers[i], s=50,
                             ax=ax, zorder=2, add_colorbar=False,
                             transform=ccrs.PlateCarree())

        # Legend entry
        start_z = representative.z.isel(time=0).item()
        end_z = representative.z.isel(time=-1).item()

        patch = mlines.Line2D([], [], color='k',
                              markerfacecolor=cmap_altitude(norm_altitude(start_z)),
                              marker=markers[i], linestyle='-', markersize=6,
                              markeredgecolor='w',
                              label=f"{start_z:.1f}–{end_z:.1f} km")
        particle_patches.append(patch)

        # Annotate last representative particle point
        rep_time_str = pd.to_datetime(representative.time[-1].item()).strftime('%Y-%m-%d %H:%M UTC')
        rep_lon = representative.lon.isel(time=-1).item()
        rep_lat = representative.lat.isel(time=-1).item()
        ax.annotate(rep_time_str, xy=(rep_lon, rep_lat), xytext=(rep_lon - 5, rep_lat - 5),
                    textcoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                    arrowprops=dict(arrowstyle="->", color='black', linewidth=1.5),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black",
                              alpha=0.8), fontsize=10, ha='left', transform=ccrs.PlateCarree())

    # Add orbit legend
    if orbit is not None:
        particle_patches.append(mlines.Line2D([], [], color='black', linestyle='-',
                                              linewidth=1.6, label="Falcon 9"))

    # KDE plot of endpoint density
    time_endpoints = trajectories.time[-1]
    # t_slice = slice(t_end + pd.Timedelta(hours=.005), t_end)

    print(f"Computing KDE of endpoints for time slice {time_endpoints.values} ...")
    points = np.vstack([
        trajectories.sel(time=time_endpoints).lon.values.flatten(),
        trajectories.sel(time=time_endpoints).lat.values.flatten()
    ]).T

    if points.shape[0] == 0:
        raise ValueError("No trajectory data found in the specified time range.")

    # Fit a Gaussian Mixture Model to the endpoint distribution
    gmm = GaussianMixture(
        n_components=3,  # Number of Gaussian components
        covariance_type='full',
        n_init=10,  # Run the EM algorithm 10 times with different initializations
        random_state=42  # Ensures these 10 initializations are the same each time
    )
    gmm.fit(points)

    lon_grid, lat_grid = np.meshgrid(
        np.linspace(max(trajectories.lon.min(), map_extent[0]),
                    min(trajectories.lon.max(), map_extent[1]), 500),
        np.linspace(max(trajectories.lat.min(), map_extent[2]),
                    min(trajectories.lat.max(), map_extent[3]), 500)
    )
    grid_points = np.c_[lon_grid.ravel(), lat_grid.ravel()]
    density = np.exp(gmm.score_samples(grid_points)).reshape(lon_grid.shape)

    # Normalize to %
    dx = np.abs(lon_grid[0, 1] - lon_grid[0, 0])
    dy = np.abs(lat_grid[1, 0] - lat_grid[0, 0])
    area_element = dx * dy
    density *= 100.0 / np.nansum(density) / area_element  # Normalize to percentage

    print("Probability density total = ", area_element * density.sum(), "%")

    lower_bound = np.percentile(density, 90)
    density_masked = np.ma.masked_less(density, lower_bound)

    cs = ax.contourf(lon_grid, lat_grid, density_masked, levels=21,
                     cmap=custom_viridis(), alpha=0.9,
                     transform=ccrs.PlateCarree(), extend='max')
    # This is a fix for the white lines between contour levels
    for c in cs.collections:
        c.set_edgecolor("face")

    # Legend
    leg = ax.legend(handles=particle_patches, loc='upper right', frameon=True)
    leg.get_frame().set_alpha(0.85)
    leg.get_frame().set_linewidth(0.4)

    # Title
    draw_title = False
    if draw_title:
        t_start_str = pd.to_datetime(trajectories.time[0].item()).strftime('%Y-%m-%d %H:%M:%S UTC')
        ax.set_title(f"Lithium back-trajectories initialized on {t_start_str}", fontsize=12, pad=10)
    else:
        ax.set_title("")

    # Adding color-bars
    gs = fig.add_gridspec(2, 2, height_ratios=[20, 1], hspace=0.1, bottom=0.12)
    ax.set_position(gs[0, :].get_position(fig))
    ax.set_subplotspec(gs[0, :])

    # Colorbar for endpoint density
    print(lower_bound, density.max())

    cax_pdf = fig.add_axes([0.1326, 0.1, 0.36, 0.042])
    ticks_pdf = MaxNLocator(nbins=5, prune=None).tick_values(lower_bound, density.max())

    cbar_pdf = fig.colorbar(cs, cax=cax_pdf, orientation='horizontal',
                            ticks=ticks_pdf, extend='max')
    cbar_pdf.set_label('Endpoint Probability (%)', fontsize=10)
    # cbar_pdf.ax.set_xticklabels([f"{t:.2f}" for t in ticks_pdf])
    cbar_pdf.ax.tick_params(labelsize=8, direction='out', length=3, width=0.5)

    # Colorbar for altitude
    cax_alt = fig.add_axes([0.53, 0.1, 0.36, 0.042])
    sm_alt = plt.cm.ScalarMappable(cmap=cmap_altitude, norm=norm_altitude)
    sm_alt.set_array([])
    cbar_alt = fig.colorbar(sm_alt, cax=cax_alt, orientation='horizontal',
                            ticks=np.linspace(96, 104, 5), extend='both')
    cbar_alt.set_label('Altitude (km)', fontsize=10)
    cbar_alt.ax.tick_params(labelsize=8, direction='out', length=3, width=0.5)

    plt.show()
    if figure_name:
        fig.savefig(f"{figure_name}.pdf", dpi=300, bbox_inches='tight')
    else:
        return fig


def plot_orbit_and_ensemble_3d(trajectories, orbit,
                               max_ensemble=100, particle_subset=None,
                               horiz_tol_km=5, vert_tol_km=1,
                               figsize=(9, 7), elev=20, azim=45, figure_name=None):
    """
    Plots the 3D position of the orbital path and the ensemble trajectories over a time range.
    Highlights intersecting trajectories and annotates them.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    orbit = orbit.set_index("timestamp")
    traj = trajectories.sortby("time")

    times = traj.time
    if len(times) == 0:
        raise ValueError("No trajectory data found in the specified time range.")

    particles = particle_subset or traj.particle.values
    ensemble = min(traj.ensemble.size, max_ensemble)
    traj = traj.sel(particle=particles, ensemble=slice(0, ensemble))

    # Compute intersections
    intersect_mask, intersect_indices = compute_intersections(
        traj, orbit, horiz_tol_km, vert_tol_km
    )

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)

    # === Plot orbit path ===
    ax.plot(orbit["GLon"], orbit["GLat"], orbit["GAlt"],
            label="Re-entry Trajectory", color="black", linewidth=2, zorder=5)

    ax.plot(orbit["GLon"], orbit["GLat"], zs=0, zdir='z',
            color='gray', linewidth=1, linestyle='--', label="Ground Track", zorder=3)

    # === Plot trajectories ===
    for p in particles:
        for e in range(ensemble):
            lon = traj.lon.sel(particle=p, ensemble=e)
            lat = traj.lat.sel(particle=p, ensemble=e)
            alt = traj.z.sel(particle=p, ensemble=e)

            # Isolate intersection points
            intersect = intersect_mask.sel(particle=p, ensemble=e)

            if intersect.any():
                ax.plot(lon, lat, alt, color='blue', alpha=0.15, linewidth=0.5, zorder=1)
                ax.scatter(lon.where(intersect, drop=True),
                           lat.where(intersect, drop=True),
                           alt.where(intersect, drop=True),
                           color='red', s=15,
                           label='Intersection' if (p == particles[0] and e == 0) else "",
                           zorder=10)
            else:
                ax.plot(lon, lat, alt, color='blue',
                        alpha=0.04, linewidth=0.6, zorder=1)

    # === Labels and title ===
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_zlabel("Altitude (km)")
    ax.set_zlim(0, 120)

    time_start = np.datetime_as_string(times[0].values, unit='s')
    time_end = np.datetime_as_string(times[-1].values, unit='s')
    ax.set_title(f"Ensemble Trajectories and Re-entry Track\n{time_start} to {time_end}",
                 fontsize=11)

    # === Annotate intersections ===
    if intersect_indices.size > 0:
        # Build table with intersect time, particle, ensemble
        intersect_df = pd.DataFrame({
            "intersect_time": trajectories.time.values[intersect_indices[:, 0]],
            "particle": intersect_indices[:, 1],
            "ensemble": intersect_indices[:, 2],
        }).sort_values(by="intersect_time")

        # Format preview string for annotation
        preview = "\n".join(
            f"P{row.particle}, E{row.ensemble}: "
            f"{pd.to_datetime(row.intersect_time).strftime('%H:%M:%S')}"
            for _, row in intersect_df.head(6).iterrows()
        )
        if len(intersect_df) > 20:
            preview += "\n..."

        ax.text2D(0.015, 0.95,
                  f"Intersections ({len(intersect_df)} total):\n" + preview,
                  transform=ax.transAxes, color='red', fontsize=9, va='top')

    else:
        ax.text2D(0.015, 0.95, "No intersections detected", transform=ax.transAxes,
                  color='gray', fontsize=9, va='top')

    # === Legend and layout ===
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.tight_layout()
    if figure_name:
        fig.savefig(figure_name, dpi=300)
    plt.show()


# Re-define the animation function after state reset
def animate_trajectories(
        trajectories, wind, orbit,
        output_dir: str,
        time_step_minutes: int = 10
):
    """
    Animate ensemble trajectories and re-entry path over time by generating individual frames.

    Parameters:
        trajectories : xr.Dataset
            Ensemble back-trajectories with dimensions (time, particle, ensemble)
        wind : xr.Dataset
            Wind field with (time, z, lat, lon) or similar dims
        orbit : pd.DataFrame
            Re-entry trajectory with 'timestamp', 'GLon', 'GLat', 'GAlt'
        output_dir : str
            Directory to save the output frames (e.g., for ffmpeg)
        time_step_minutes : int
            Time step (in minutes) between animation frames
    """
    os.makedirs(output_dir, exist_ok=True)

    start_time = pd.to_datetime(trajectories.time.min().values)
    end_time = pd.to_datetime(trajectories.time.max().values)

    times = pd.date_range(start=start_time, end=end_time, freq=f"{time_step_minutes}min")

    for i, t in enumerate(times):
        print(f"Generating frame {i + 1}/{len(times)} at {t}")
        # Slice backwards in time (since these are back-trajectories)
        traj_slice = trajectories.sel(time=slice(trajectories.time.max().values, t))
        orbit_slice = orbit[orbit["timestamp"] <= t]

        # Skip if the slice is empty
        if traj_slice.time.size == 0:
            print(f"  Skipped frame at {t} — no data in time slice.")
            continue

        frame_path = os.path.join(output_dir, f"frame_{i:03d}.png")

        try:
            fig = visualize_trajectories_percentile_kde(
                trajectories=traj_slice,
                wind=wind,
                orbit=orbit_slice,
                figure_name=None
            )

            fig.savefig(frame_path, dpi=300, bbox_inches='tight')
            fig.close()

        except Exception as e:
            print(f"  Error generating frame at {t}: {e}")
