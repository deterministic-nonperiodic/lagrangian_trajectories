import os
import numpy as np
import xarray as xr
import pandas as pd
from dateutil import parser

from scipy.integrate import solve_ivp

from pyproj import Transformer

# Use an equal-area projection (Eckert IV) to convert (lat, lon) to meters
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


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

    # Helper function to convert duration or timestep to seconds
    def convert_to_seconds(value, unit='s'):
        if isinstance(value, (int, float)):
            return pd.Timedelta(value, unit=unit).total_seconds()
        elif isinstance(value, str):
            return pd.Timedelta(value).total_seconds()
        else:
            raise ValueError(f"Unknown format for argument '{value}'")

    # Convert both duration and timestep to seconds
    duration = convert_to_seconds(duration)
    timestep = convert_to_seconds(timestep, 's')

    # Compute the number of steps
    num_steps = int(abs(duration) / abs(timestep)) + 1  # Include zero

    direction = np.sign(timestep) * np.sign(duration)

    return np.linspace(0, direction * abs(duration), num_steps)


class LagrangianTrajectories:
    def __init__(self, data, timestep=None, integration_method="RK23", start_time=None,
                 interpolation_method="linear", verbose=False, ensemble=False):
        """
        Initialize the Lagrangian trajectory calculator.

        Parameters:
        - wind_data: xarray.Dataset containing 'u', 'v', 'w' wind components
          with dimensions ('time', 'z', 'lat', 'lon')
        - dt: Time step in seconds (default: 60s)
        - integration_method: Integration method, one of ['RK23', 'RK45', 'DOP853', 'LSODA'] (default: 'RK2')
        - interpolation_method: Interpolation method, one of ['linear', 'nearest']
        - start_time: Start time for the simulation in format 'yyyy-mm-ddTHH:MM:SS'
        - verbose: Print maximum wind velocity at each time step
        - ensemble: Compute ensemble trajectories (default: False)
        """

        self.verbose = verbose

        self.wind = data[["u", "v", "w"]]

        if timestep is None:
            # estimate from dataset and convert to seconds
            timestep = np.median(np.diff(self.wind.time.values)) / np.timedelta64(1, 's')

        self.timestep = timestep
        self.method = integration_method
        self.interp_method = interpolation_method

        time = self.wind.time.values

        if start_time is None:
            self.start_time = time[0]
        else:
            start_time = parser.parse(start_time)
            self.start_time = pd.to_datetime(start_time).to_numpy()

        # Compute relative time in seconds
        self.rel_time = (time - self.start_time).astype("timedelta64[ns]") / np.timedelta64(1, 's')

        # Convert lat/lon to curvilinear x/y in meters using the map projection
        x_mesh, y_mesh = transformer.transform(
            *np.meshgrid(data.lon.values, data.lat.values, indexing="ij")
        )

        # Adjust wind components
        # self.wind['u'] = self.wind['u'] / np.cos(np.deg2rad(self.wind.lat))

        # Replace lon lat with curvilinear coordinates in meters
        self.wind = self.wind.assign_coords(
            lon=x_mesh[:, 0],
            lat=y_mesh[0, :],
            rel_time=self.rel_time)

    def velocity(self, time, state):
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

        if self.verbose:
            print(time,
                  "Max u: {:4.4f}".format(wind.u.max().values), ' ',
                  "Max w: {:0.4f}".format(wind.w.max().values)
                  )

        # earth_radius = 6.3712e6  # Radius of Earth (m)
        u_mp = wind.u.data  # / (earth_radius * np.cos(np.deg2rad(wind.lat.data)))
        v_mp = wind.v.data  # / earth_radius

        return np.column_stack([u_mp, v_mp, wind.w.data])

    def advect_particles(self, initial_positions, duration=None):
        """
        Compute trajectories for a set of particles.

        Parameters:
        - initial_positions: list of (lat, lon, z) tuples
        - duration: Length of the simulation in time units (default: None)

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

        # Solve the ODE system
        initial_positions = np.array(initial_positions)
        num_particles, state_size = initial_positions.shape[:2]

        # initialize trajectories array
        trajectories = np.zeros((num_particles, state_size, times.size))

        for i, init_pos in enumerate(initial_positions):
            # transform initial position to meters: Transpose lat/lon
            init_pos[:2] = transformer.transform(*init_pos[:2])

            # Solve the ODE system for one particle. We are solving dx/dt = v(t, x, y, z)
            trajectories[i] = solve_ivp(self.velocity, t_span=time_span, y0=init_pos,
                                        t_eval=times, method=self.method).y

        # transform back to lat/lon
        trajectories[:, 0], trajectories[:, 1] = transformer.transform(trajectories[:, 0],
                                                                       trajectories[:, 1],
                                                                       direction='INVERSE')

        # transform time back to datetime
        times = pd.to_datetime(times, origin=self.start_time, unit='s')

        # Create xarray dataset with trajectories
        trajectories = xr.Dataset(
            {
                "lon": (("particle", "time"), trajectories[:, 0]),
                "lat": (("particle", "time"), trajectories[:, 1]),
                "z": (("particle", "time"), 1e-3 * trajectories[:, 2])
            },
            coords={"particle": np.arange(num_particles), "time": times}
        )
        trajectories['z'].attrs['standard_name'] = 'geometric height'
        trajectories['z'].attrs['units'] = 'kilometers'

        return trajectories


def visualize_trajectories(trajectories, wind_data, start_time='', fig_name="trajectories.png"):
    # Plot one trajectory
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Create figure and axis with geographic projection
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True,
                           subplot_kw={"projection": ccrs.PlateCarree()})
    # ax.coastlines()
    ax.set_extent([-30, 30, 40, 75], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    ax.add_feature(cfeature.LAND, color='gray', alpha=0.2)

    ax.add_feature(cfeature.COASTLINE, linewidth=0.25)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True

    # visualize mean wind streamlines
    altitude = trajectories.sel(particle=0).z.isel(time=0).values

    wind_data = wind_data.sel(time=slice(trajectories.time[-1], trajectories.time[0]))
    wind_data = wind_data.sel(z_mc=1e3 * altitude, method='nearest').mean(dim='time')

    wind_data.plot.streamplot(x="lon", y="lat", u="u", v="v",
                              color='black', ax=ax, transform=ccrs.PlateCarree(),
                              density=1.6, linewidth=0.15)

    # plot paths
    markers = ['o', 's', 'd', '^', 'v', '<', '>', 'p', 'h', 'D', 'P', 'X']

    v_min = trajectories["z"].min().values
    v_max = trajectories["z"].max().values

    for i in range(trajectories.particle.size):
        # Plot trajectory line
        ax.plot(trajectories.lon[i], trajectories.lat[i],
                color='k', alpha=0.6, linewidth=1.6, transform=ccrs.PlateCarree())
        # Get starting altitude point
        particle = trajectories.sel(particle=i)

        start_altitude = np.round(particle.z.isel(time=0).values, 1)

        particle.plot.scatter(x="lon", y="lat", hue='z',
                              # color=default_colors[i],
                              edgecolors='w', s=50, marker=markers[i],
                              add_colorbar=(i == 0), vmin=v_min, vmax=v_max, cmap='jet',
                              cbar_kwargs={"label": "altitude / km", 'pad': 0.02,
                                           "shrink": 0.6, "extend": "both"},
                              label="Trajectory starting @ {} km".format(start_altitude),
                              ax=ax, transform=ccrs.PlateCarree())

    ax.legend(loc='upper right', frameon=True)

    ax.set_title(f"Lagrangian Back-trajectories {' '.join(start_time.split('T'))} UTC")
    plt.show()

    fig.savefig(fig_name, dpi=300)


if __name__ == "__main__":
    # Load simulated 3D wind field (example)
    base_path = "/home/deterministic-nonperiodic/IAP/Experiments/vortex/data/FALCON/"
    filename = os.path.join(base_path, "UA-ICON_NWP_atm_ML_DOM01_falcon2_20250219-20.nc")

    wind_data = xr.open_dataset(filename)  # Dataset containing 'u', 'v', 'w'
    # wind_data = wind_data.chunk(time=10, z_mc=10, z_ifc=10, lat=111, lon=111)

    # Apply radar correction: time delay of 50 minutes
    lagged_time = wind_data.time + pd.Timedelta(minutes=-0)
    wind_data = wind_data.assign_coords(time=lagged_time)

    # Define start time with format yyyy-mm-ddTHH:MM:SS
    start_time = "2025-02-20T00:30:00"

    # Define initial particle positions (lat, lon, z[meters]):
    initial_positions = [(11.46, 54.07, 95.9e3), (11.46, 54.07, 95.75e3)]
    #
    # Define simulation duration and time step. It can be negative for backward trajectories
    time_step = "10 minutes"  # Time step, flexible units [min, m, s, sec, h, hours, ...]
    duration = "-20.75 hours"  # Duration of the simulation

    solver_method = 'LSODA'  # options: ['RK23', 'RK45', 'DOP853', 'LSODA']
    interp_method = 'linear'  # options: ['linear', 'nearest', 'cubic', 'quadratic']

    # Initialize the trajectory calculator: Using Runge-Kutta 2nd order method
    solver = LagrangianTrajectories(wind_data, timestep=time_step, start_time=start_time,
                                    integration_method=solver_method,
                                    interpolation_method=interp_method,
                                    verbose=True, ensemble=False)

    # Compute trajectories
    trajectories = solver.advect_particles(initial_positions, duration=duration)

    # Store file as netcdf
    filename = f"trajectories_{solver_method}_{interp_method}_{start_time}.nc"
    trajectories.to_netcdf(os.path.join(base_path, filename))

    # Visualize trajectories
    fig_name = (f"/home/deterministic-nonperiodic/rc_trajectories_{solver_method}"
                f"_{interp_method}_{start_time}.png")

    visualize_trajectories(trajectories, wind_data, fig_name=fig_name, start_time=start_time)
