import os
import numpy as np
import xarray as xr
import pandas as pd
from dateutil import parser
from functools import partial
from itertools import product

from scipy.integrate import solve_ivp
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

from tqdm import tqdm
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as creature

import alphashape
from pyproj import Transformer
from shapely.geometry import Point

# Use an equal-area projection (Eckert IV) to convert (lat, lon) to meters
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def custom_viridis():
    # Get the viridis colormap
    viridis = plt.cm.get_cmap('viridis', 256)

    # Convert to array of colors
    viridis_colors = viridis(np.linspace(0, 1, 256))

    # Create new color map that fades in from white
    white_to_start = 100
    white_to_viridis = np.vstack([
        np.linspace([1, 1, 1, 1], viridis_colors[white_to_start], white_to_start),
        viridis_colors[white_to_start:]  # rest of viridis
    ])

    return mcolors.LinearSegmentedColormap.from_list("custom_viridis", white_to_viridis)


# Helper function to convert duration or timestep to seconds
def convert_to_seconds(value, unit='s'):
    if isinstance(value, (int, float)):
        return pd.Timedelta(value, unit=unit).total_seconds()
    elif isinstance(value, str):
        return pd.Timedelta(value).total_seconds()
    elif np.issubdtype(value.dtype, np.timedelta64):
        return value / np.timedelta64(1, 's')
    else:
        raise ValueError(f"Unknown format for argument '{value}'")


def generate_eval_time(duration, timestep):
    """
    Generate an array of evaluation times starting at 0 and incrementing by a specified timestep.

    :param duration: Total duration (can be a string, int, or float).
    :param timestep: Time step (can be a string, int, or float). It can be negative for backward time progression.
    :return: A numpy array of evaluation times in seconds.
    :raises ValueError: If 'duration' or 'timestep' are in an unknown format.
    """

    # Return empty array if duration is None
    if duration is None:
        return np.zeros(1)

    # Convert both duration and timestep to seconds
    duration = convert_to_seconds(duration)
    timestep = convert_to_seconds(timestep, 's')

    # Compute the number of steps
    num_steps = int(abs(duration) / abs(timestep)) + 1  # Include zero

    direction = np.sign(timestep) * np.sign(duration)

    return np.linspace(0, direction * abs(duration), num_steps)


class LagrangianTrajectories:
    def __init__(self, data, timestep=None, integration_method="RK23", start_time=None,
                 interpolation_method="linear", verbose_level=None, time_lag=None):
        """
        Initialize the Lagrangian trajectory calculator.

        Parameters:
        - wind_data: xarray.Dataset containing 'u', 'v', 'w' wind components
          with dimensions ('time', 'z', 'lat', 'lon')
        - dt: Time step in seconds (default: 60s)
        - integration_method: Integration method. one of ['RK23', 'RK45', 'DOP853', 'LSODA']
                              (default: 'RK2')
        - interpolation_method: Interpolation method, one of ['linear', 'nearest']
        - start_time: Start time for the simulation in format 'yyyy-mm-ddTHH:MM:SS'
        - verbose: Print maximum wind velocity at each time step
        """

        self.verbose = verbose_level or 0

        self.wind = data[["u", "v", "w"]]

        # Check if the dataset contains the required dimensions
        if time_lag is not None:
            # Apply radar correction: time delay of 50 minutes
            self.wind = self.wind.assign_coords(time=self.wind.time - pd.Timedelta(time_lag))
            # self.wind = radar_correction(self.wind, time_lag=time_lag)

        # Extract time coordinate
        time = self.wind.time.values

        if timestep is None:
            # estimate from dataset and convert to seconds
            timestep = convert_to_seconds(np.median(np.diff(time)))

        self.timestep = timestep
        self.method = integration_method
        self.interp_method = interpolation_method

        if start_time is None:
            self.start_time = time[0]
        else:
            start_time = parser.parse(start_time)
            self.start_time = pd.to_datetime(start_time).to_numpy()

        # Compute relative time in seconds
        self.rel_time = convert_to_seconds(time - self.start_time)

        # Convert lat/lon to curvilinear x/y in meters using the map projection
        xc, yc = transformer.transform(
            *np.meshgrid(self.wind.lon.values, self.wind.lat.values, indexing="ij")
        )

        # Replace lon lat with curvilinear coordinates in meters
        self.wind = self.wind.assign_coords(lon=xc[:, 0], lat=yc[0], rel_time=self.rel_time)

    def velocity(self, time, state, noise_scale=False):
        """Compute wind velocity at given positions."""
        x, y, z = state.T  # Extract current positions

        # convert relative to absolute time
        time = pd.to_datetime(time, origin=self.start_time, unit='s')

        # Create interpolation points: (time, z, lat, lon)
        points = dict(z_mc=z, lat=y, lon=x)

        if 'z_ifc' in self.wind.dims: points['z_ifc'] = z

        # Perform interpolation. Velocity is set to zero if the particle leaves the domain
        wind = self.wind.interp(time=time, **points,
                                method=self.interp_method,
                                kwargs={'fill_value': 0.0})

        if self.verbose > 1:
            print(time,
                  "Max u: {:4.4f}".format(wind.u.max().values), ' ',
                  "Max w: {:0.4f}".format(wind.w.max().values)
                  )

        # Extract wind components
        wind_vector = np.column_stack([wind.u.data, wind.v.data, wind.w.data])

        # Add Gaussian noise to horizontal wind: magnitude noise_scale [m/s]
        if noise_scale:
            # Generate log-normal noise for each component
            wind_vector *= np.random.lognormal(mean=0.0, sigma=0.15, size=wind_vector.shape)

        return wind_vector

    def advect_particles(self, initial_positions, duration=None, ensemble_size=None):
        """
        Compute trajectories for a set of particles.

        Parameters:
        - initial_positions: list of (lat, lon, z) tuples
        - duration: Length of the simulation in time units (default: None)
        - ensemble_size: Number of ensemble members (default: None)

        Returns:
        - xarray.Dataset containing particle trajectories with dimensions ('particle', 'time')
        """

        # Get end date based on simulation duration
        # end_date = self.start_time + np.sign(self.dt) * pd.Timedelta(duration)
        times = generate_eval_time(duration, self.timestep)

        # check if time is within data bounds. clip array to valid range
        rel_time = self.rel_time.astype(float)
        times = times[(times >= np.min(rel_time)) & (times <= np.max(rel_time))]

        # calculate integration time span
        time_span = (times[0], times[-1])

        # Transform lat/lon to meters for all positions (if shape is (N, 3))
        initial_positions_m = np.array([
            (*transformer.transform(lon, lat), alt)
            for lon, lat, alt in initial_positions
        ])

        # Create ensemble run
        if ensemble_size is None:
            ensemble_size = 1

        num_particles, state_size = np.shape(initial_positions)[:2]

        # initialize trajectories array
        trajectories = np.zeros((state_size, times.size, num_particles, ensemble_size))

        # Define the function to integrate for each particle, with noise applied conditionally
        def integrate_particle(i, j):
            pos = initial_positions_m[i]

            if self.verbose > 0:
                print(f"{(j + 1) / ensemble_size} %: Integrating particle {i} in ensemble {j} for"
                      f" {duration}")

            # Solve the ODE system for the particle with the selected velocity function
            # Define the velocity function with the chosen noise scale except for the first
            # ensemble member (j = 0), no noise.
            result = solve_ivp(partial(self.velocity, noise_scale=j),
                               t_span=time_span, y0=pos,
                               t_eval=times, method=self.method).y

            # Store the result in the trajectories array, and indices for aggregating
            return i, j, result

        # Run all integrations in parallel (you can set n_jobs=-1 to use all cores)
        results = Parallel(n_jobs=-1)(
            delayed(integrate_particle)(i, j)
            for i, j in product(range(num_particles), range(ensemble_size))
        )

        # Store results in the trajectories array
        for (i, j, traj) in results:
            trajectories[..., i, j] = traj

        # transform back to lat/lon
        trajectories[:2] = transformer.transform(*trajectories[:2], direction='INVERSE')

        # transform time back to datetime
        times = pd.to_datetime(times, origin=self.start_time, unit='s')

        # Create xarray dataset with trajectories
        trajectories = xr.Dataset(
            {
                "lon": (("time", "particle", "ensemble"), trajectories[0]),
                "lat": (("time", "particle", "ensemble"), trajectories[1]),
                "z": (("time", "particle", "ensemble"), 1e-3 * trajectories[2])
            },
            coords={"particle": np.arange(num_particles),
                    "ensemble": np.arange(ensemble_size),
                    "time": times}
        )
        trajectories['z'].attrs['standard_name'] = 'geometric height'
        trajectories['z'].attrs['units'] = 'kilometers'

        return trajectories.squeeze()


def visualize_trajectories(trajectories, wind, fig_name=None):
    """Plot one trajectories on a map"""

    # Create figure and axis with geographic projection
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True,
                           subplot_kw={"projection": ccrs.PlateCarree()})
    # ax.coastlines()
    map_extend = [-30.01, 20.01, 40, 70]
    ax.set_extent(map_extend, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    ax.add_feature(cfeature.LAND, color='gray', alpha=0.25)

    ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True

    # visualize mean wind streamlines
    altitude = trajectories.sel(particle=0, method='nearest').z.isel(time=0).mean('ensemble').values

    wind = wind.sel(time=slice(trajectories.time[-1], trajectories.time[0]))
    wind = wind.sel(z_mc=1e3 * altitude, method='nearest').mean(dim='time')

    wind.plot.streamplot(x="lon", y="lat", u="u", v="v",
                         color='black', ax=ax, transform=ccrs.PlateCarree(),
                         density=1.6, linewidth=0.15)

    # Plot paths with cone/spread and probability density
    markers = ['o', 's', 'd', '^', 'v', '<', '>', 'p', 'h', 'D', 'P', 'X']

    v_min = trajectories["z"].min().values
    v_max = trajectories["z"].max().values

    # Loop over each trajectory
    particles = trajectories.particle.values
    representative_particle = trajectories.isel(ensemble=0, particle=0)

    for i in particles:
        # Get particle for the current trajectory
        particle = trajectories.sel(particle=i)

        # Separate unperturbed (ensemble=0) and perturbed members
        representative_particle = particle.sel(ensemble=0) # median('ensemble')

        # first ensemble with unperturbed winds
        ax.plot(representative_particle.lon,
                representative_particle.lat,
                color='black', alpha=0.6, linewidth=1.0, zorder=3,
                transform=ccrs.PlateCarree())

        # Stack all perturbed trajectories into numpy arrays
        ax.plot(particle.lon.values,
                particle.lat.values,
                color='#bfbfbf', alpha=0.1, linewidth=0.8, zorder=1,
                transform=ccrs.PlateCarree())

        # Get the starting and ending altitude
        start_altitude = np.round(particle.z.isel(time=0).mean('ensemble').values, 1)
        end_altitude = np.round(particle.z.isel(time=-1).mean('ensemble').values, 1)

        # Scatter plot for the main ensemble member
        sampled_particle = representative_particle.isel(time=slice(None, None, 2))
        sampled_particle.plot.scatter(x="lon", y="lat", hue='z', cmap='jet',
                                      edgecolors=None, s=60, marker=markers[i],
                                      add_colorbar=(i == particles[0]),
                                      vmin=90, vmax=104,
                                      cbar_kwargs={"label": "altitude / km",
                                                   'pad': 0.01, "shrink": 0.78,
                                                   'ticks': [90, 92, 94, 96, 98, 100, 102, 104],
                                                   "extend": "both"},
                                      zorder=2,
                                      label="Altitude range: {}-{} km".format(
                                          start_altitude, end_altitude),
                                      ax=ax, transform=ccrs.PlateCarree())

    # For example, final point of first trajectory
    last_time = pd.to_datetime(trajectories.time[-1].values).strftime('%Y-%m-%d %H:%M UTC')

    final_lon = representative_particle.isel(time=-1).lon.values
    final_lat = representative_particle.isel(time=-1).lat.values

    # noinspection PyProtectedMember
    ax.annotate(
        f"{last_time}",
        xy=(final_lon, final_lat),  # Arrow points here
        xytext=(final_lon, final_lat - 5),  # Text appears here
        textcoords=ccrs.PlateCarree()._as_mpl_transform(ax),  # Geographic transform
        arrowprops=dict(
            arrowstyle="->",
            color='black',
            linewidth=1.5
        ),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8),
        fontsize=9,
        ha='left',
        transform=ccrs.PlateCarree()
    )

    # Create ensemble probability density around the main trajectory: takes last 2 hours
    time_range = [trajectories.time[-1] + pd.Timedelta(hours=1), trajectories.time[-1]]

    points = np.vstack([
        trajectories.sel(time=slice(*time_range)).lon.values.flatten(),
        trajectories.sel(time=slice(*time_range)).lat.values.flatten()
    ]).T

    # Fit GMM density estimation
    min_samples = max(1, trajectories.ensemble.size // 4)
    db = DBSCAN(eps=0.482, min_samples=min_samples)

    labels = db.fit_predict(points)

    num_clusters = 2  # min(len(set(labels)) - (1 if -1 in labels else 0), 4)
    print(f"Estimated number of clusters: {num_clusters}")

    gmm = GaussianMixture(n_components=num_clusters, covariance_type='full')
    gmm.fit(points)

    # Create a grid to evaluate the KDE on
    lon_extend = [trajectories.lon.min().values, trajectories.lon.max().values]
    lat_extend = [trajectories.lat.min().values, trajectories.lat.max().values]

    lon_grid, lat_grid = np.meshgrid(np.linspace(*lon_extend, 250),
                                     np.linspace(*lat_extend, 250))

    grid_points = np.dstack((lon_grid, lat_grid)).reshape(-1, 2)

    # Evaluate the density on the grid
    density = np.exp(gmm.score_samples(grid_points)).reshape(lon_grid.shape)

    # Normalize over grid (assumes lon and lat are 2D arrays from meshgrid)
    area_element = np.abs(lon_grid[0, 1] - lon_grid[0, 0]) * np.abs(lat_grid[1, 0] - lat_grid[0, 0])

    # Total "mass" of density
    density = 100 * density / (np.sum(density) * area_element)

    alpha_shape = alphashape.alphashape(points, alpha=0.05)  # smaller alpha = more concave

    # Create a mask for points outside the alpha shape
    mask = np.array([
        alpha_shape.contains(Point(x, y))
        for x, y in zip(lon_grid.ravel(), lat_grid.ravel())
    ]).reshape(lon_grid.shape)

    # Mask the KDE density
    density = np.ma.masked_where(~mask, density)

    print(density.max(), density.min())

    # Plot the density with contours
    cs = ax.contourf(lon_grid, lat_grid, density,
                     levels=np.linspace(0, 20, 20), zorder=0,
                     vmin=0, vmax=20,
                     cmap=custom_viridis(), alpha=0.99, transform=ccrs.PlateCarree())

    fig.colorbar(cs, ax=ax, orientation='horizontal', label='Probability Density (%)',
                 ticks=np.arange(0, 80, 2), extend='right', shrink=0.7, pad=0.025)

    # Add legend for markers
    ax.legend(loc='upper right', frameon=True)

    start_time_str = pd.to_datetime(trajectories.time[0].values).strftime('%Y-%m-%d %H:%M UTC')

    ax.set_title(f"Back-trajectories initialized on {start_time_str}")

    plt.show()

    if fig_name is not None:
        fig.savefig(fig_name, dpi=300)


if __name__ == "__main__":
    # Load simulated 3D wind field (example)
    base_path = "/home/deterministic-nonperiodic/IAP/Experiments/vortex/data/FALCON/"
    filename = os.path.join(base_path, "UA-ICON_NWP_atm_ML_DOM01_falcon2_20250219-20.nc")

    wind_data = xr.open_dataset(filename)  # Dataset containing 'u', 'v', 'w'

    # Define start time with format yyyy-mm-ddTHH:MM:SS
    start_time = "2025-02-20T00:30:00"

    # Define initial particle positions (lat, lon, z[meters]):
    initial_positions = [(11.46, 54.07, 95.9e3), (11.46, 54.07, 95.75e3)]

    # Define time step. It can be negative for backward trajectory calculation
    time_step = "-10 minutes"  # Time step with flexible units [min, m, s, sec, h, hours, etc...]
    duration = "20:47:13"  # Duration of the simulation

    time_lag = "0minutes"  # Time lag for radar correction

    solver_method = 'RK23'  # options: ['RK23', 'RK45', 'DOP853', 'LSODA']
    interp_method = 'linear'  # options: ['linear', 'nearest', 'cubic', 'quadratic']

    ensemble_size = 500  # Number of ensemble members
    
    # Initialize the trajectory calculator: Using Runge-Kutta 2nd order method
    solver = LagrangianTrajectories(wind_data, timestep=time_step, start_time=start_time,
                                    integration_method=solver_method, time_lag=time_lag,
                                    interpolation_method=interp_method, verbose_level=1)

    # Compute trajectories
    trajectories_dataset = solver.advect_particles(initial_positions, duration=duration,
                                                   ensemble_size=ensemble_size)
    
    # Store file as netcdf
    filename = f"trajectories_{solver_method}_{interp_method}_{start_time}.nc"
    trajectories.to_netcdf(os.path.join(base_path, filename))

    # Visualize trajectories
    fig_name = (f"/home/deterministic-nonperiodic/rc_trajectories_{solver_method}"
                f"_{interp_method}_{start_time}_lag{time_lag}.png")

    visualize_trajectories(trajectories_dataset, wind_data, fig_name=fig_name)
