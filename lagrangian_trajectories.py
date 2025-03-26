import os
import numpy as np
import xarray as xr
import pandas as pd
from mpmath import extend
from scipy.integrate import solve_ivp

from pyproj import Transformer

# Use an equal-area projection (Eckert IV) to convert (lat, lon) to meters
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


class LagrangianTrajectories:
    def __init__(self, data, dt=60, integration_method="RK23", start_time=None,
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
        self.wind = data[["u", "v", "w"]]
        self.dt = dt
        self.method = integration_method
        self.interp_method = interpolation_method

        time = self.wind.time.values

        self.verbose = verbose

        if start_time is None:
            self.start_time = time[0]
        else:
            self.start_time = pd.to_datetime(start_time).to_numpy()

        # Compute relative time in seconds
        self.rel_time = 1e-9 * (time - self.start_time).astype("timedelta64[ns]")

        # Convert lat/lon to curvilinear x/y in meters using the map projection
        x_mesh, y_mesh = transformer.transform(
            *np.meshgrid(data.lon.values, data.lat.values, indexing="ij")
        )

        # Replace lon lat with curvilinear coordinates in meters
        self.wind = self.wind.assign_coords(lon=x_mesh[:, 0],
                                            lat=y_mesh[0, :],
                                            rel_time=self.rel_time)

    def velocity(self, time, state):
        """Compute wind velocity at given positions."""
        x, y, z = state.T  # Extract current positions

        # convert relative to absolute time
        time = pd.to_datetime(time, origin=self.start_time, unit='s')

        # Create interpolation points: (time, z, lat, lon)
        points = dict(z_mc=z, z_ifc=z, lat=y, lon=x)

        # Perform interpolation. Velocity is set to zero if the particle leaves the domain
        wind = self.wind.interp(time=time, **points, method=self.interp_method,
                                kwargs={'fill_value': 0.0})

        if self.verbose:
            print(time, "Max u: ", wind.u.max().values, " Max w: ", wind.w.max().values)

        return np.column_stack([wind.u.data, wind.v.data, wind.w.data])

    def advect_particles(self, initial_positions, num_steps):
        """
        Compute trajectories for a set of particles.

        Parameters:
        - initial_positions: list of (lat, lon, z) tuples
        - num_steps: Number of integration steps

        Returns:
        - xarray.Dataset containing particle trajectories with dimensions ('particle', 'time')
        """
        num_particles = len(initial_positions)
        times = np.arange(0, num_steps * self.dt, self.dt)

        # check if time is within data bounds. clip array to valid range
        rel_time = self.rel_time.astype(float)
        times = times[(times >= np.min(rel_time)) & (times <= np.max(rel_time))]

        # calculate integration time span
        time_span = (times[0], times[-1])

        # Define integration function: We are solving dx/dt = v(t, x, y, z)
        def state_func(t, state):
            return self.velocity(t, state)

        # Solve the ODE system
        initial_positions = np.array(initial_positions)
        state_size = initial_positions.shape[1]

        # initialize trajectories array
        trajectories = np.zeros((num_particles, state_size, times.size))

        for i, init_pos in enumerate(initial_positions):
            # transform initial position to meters: Transpose lat/lon
            init_pos[:2] = transformer.transform(*init_pos[:2])

            print(init_pos)

            # Solve the ODE system for one particle
            result = solve_ivp(state_func, time_span, y0=init_pos, method=self.method, t_eval=times)

            trajectories[i] = result.y

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
                "z": (("particle", "time"), trajectories[:, 2])
            },
            coords={"particle": np.arange(num_particles), "time": times}
        )

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
    wind_data = wind_data.sel(z_mc=altitude, method='nearest').mean(dim='time')

    wind_data.plot.streamplot(x="lon", y="lat", u="u", v="v",
                              color='black', ax=ax, transform=ccrs.PlateCarree(),
                              density=1.6, linewidth=0.15)

    # plot paths
    markers = ['o', 's', 'd', '^', 'v', '<', '>', 'p', 'h', 'D', 'P', 'X']

    v_min = 1e-3 * trajectories["z"].min().values
    v_max = 1e-3 * trajectories["z"].max().values

    for i in range(trajectories.particle.size):
        # Plot trajectory line
        ax.plot(trajectories.lon[i], trajectories.lat[i],
                color='k', alpha=0.6, linewidth=1.6, transform=ccrs.PlateCarree())
        # Get starting altitude point
        particle = trajectories.sel(particle=i)
        particle['z'] = 1e-3 * particle['z']  # convert to km

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
    filename = os.path.join(base_path, "UA-ICON_NWP_atm_ML_DOM01_falcon_20250219-20.nc")

    wind_data = xr.open_dataset(filename)  # Dataset containing 'u', 'v', 'w'

    # Apply radar correction: time delay of 50 minutes
    lagged_time = wind_data.time + pd.Timedelta(minutes=-50)
    wind_data = wind_data.assign_coords(time=lagged_time)

    # Define start time with format yyyy-mm-ddTHH:MM:SS
    start_time = "2025-02-19T22:00:00"

    # Define initial particle positions (lat, lon, z[meters]): Kuehlungsborn, Germany at 95.75 km
    initial_positions = [(11.77, 54.12, 95.9e3), (11.77, 54.12, 95.75e3), ]

    # Define time step in seconds. It can be negative for backward trajectory calculation
    time_step = -600  # 30 minutes
    num_steps = 300  # number of time steps

    solver_method = 'RK45'  # options: ['RK23', 'RK45', 'DOP853', 'LSODA']
    interp_method = 'linear'  # options: ['linear', 'nearest', 'cubic', 'quadratic']

    # Initialize the trajectory calculator: Using Runge-Kutta 2nd order method
    solver = LagrangianTrajectories(wind_data, dt=time_step, start_time=start_time,
                                    integration_method=solver_method,
                                    interpolation_method=interp_method,
                                    verbose=False, ensemble=False)

    # Compute trajectories
    trajectories = solver.advect_particles(initial_positions, num_steps=num_steps)

    # Store file as netcdf
    filename = f"back_trajectories_{solver_method}_{interp_method}_{start_time}.nc"
    trajectories.to_netcdf(os.path.join(base_path, filename))

    # Visualize trajectories
    fig_name = (f"/home/deterministic-nonperiodic/rc_trajectories_{solver_method}"
                f"_{interp_method}_{start_time}.png")

    visualize_trajectories(trajectories, wind_data, fig_name=fig_name, start_time=start_time)
