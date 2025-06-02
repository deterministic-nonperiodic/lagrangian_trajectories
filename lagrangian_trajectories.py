import os
import dask
import numpy as np
import xarray as xr
import pandas as pd
from dateutil import parser
from functools import partial

from scipy.integrate import solve_ivp

from tools import convert_to_seconds, save_cf_compliant
from tools import generate_eval_time, read_falcon, sigma_components, generate_mean_wind

from pyproj import Transformer

from joblib import Parallel, delayed
from visualization import plot_orbit_and_ensemble_3d, visualize_trajectories_percentile_kde

# Use an equal-area projection (Eckert IV) to convert (lat, lon) to meters
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# Adjust memory size per chunk
dask.config.set({"array.chunk-size": "256 MiB"})  # should fit in 256 GB RAM


class LagrangianTrajectories:
    def __init__(self, data, timestep=None, integration_method="RK23", start_time=None,
                 interpolation_method="linear", noise_type=None,
                 verbose_level=None, time_lag=None):
        """
        Initialize the Lagrangian trajectory calculator.

        Parameters:
        - wind_data: xarray.Dataset containing 'u', 'v', 'w' wind components
          with dimensions ('time', 'z', 'lat', 'lon')
        - timestep: Time step in seconds (default: 60s)
        - integration_method: Integration method ['RK23', 'RK45', 'DOP853', 'LSODA']
                              (default: 'RK2')
        - interpolation_method: Interpolation method, one of ['linear', 'nearest']
        - start_time: Start time for the simulation in format 'yyyy-mm-ddTHH:MM:SS'
        - verbose: Print maximum wind velocity at each time step
        """

        self.verbose = verbose_level or 0

        self.components = ['u', 'v', 'w']
        self.wind = data[self.components]

        self.state_size = len(self.components)

        # free up memory
        del data

        # Check if the dataset contains the required dimensions
        self.time_lag = time_lag
        if time_lag is not None:
            # Apply radar correction: time delay of 50 minutes
            self.wind = self.wind.assign_coords(time=self.wind.time - pd.Timedelta(self.time_lag))

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

        # Create a function to generate noise
        self.noise_type = noise_type or 'lognormal'  # Default noise scale

        if self.noise_type == 'lognormal':
            self.mean_speed = generate_mean_wind(self.wind)

        # Chunk the wind data for better performance
        # self.wind = self.wind.chunk("auto")  # Automatically chunk the data for performance
        print("working with chunked data: ", self.wind.chunksizes)

        # Set the variability of the wind field
        self.sigma = sigma_components()

        # Convert lat/lon to curvilinear x/y in meters using the map projection
        latitude = self.wind.lat.values
        longitude = self.wind.lon.values

        xc, yc = transformer.transform(*np.meshgrid(longitude, latitude, indexing="ij"))

        # Replace lon lat with curvilinear coordinates in meters
        self.wind = self.wind.assign_coords(lon=xc[:, 0], lat=yc[0], rel_time=self.rel_time)

        # Set coordinate for interpolation
        self.interp_dims = [
            dim for dim in ['time', 'lat', 'lon', 'z_mc', 'z_ifc'] if dim in self.wind.dims
        ]

    def lognormal_noise_generator(self, z, seed: int = None, max_sigma=0.5):
        """
        Generate additive normal noise for wind components (u, v, w)
        with altitude-dependent standard deviations.

        Parameters:
            z : np.ndarray
                1D array of altitudes at which to compute noise.
            seed : int, optional
                Seed for reproducibility.
            max_sigma : float, optional
                Maximum sigma for lognormal distribution.

        Returns:
            np.ndarray:
                Array of shape (3, len(z)) containing noise for u, v, and w.
        """
        rng = np.random.default_rng(seed)

        kwargs = dict(coords={'z_mc': z}, method='linear', kwargs={'fill_value': "extrapolate"})

        sigma_rel = self.sigma.interp(**kwargs) / self.mean_speed.interp(**kwargs)

        sigma_rel = sigma_rel.to_array().values.clip(0, max_sigma)  # shape: (3, len(z))

        # Ensure zero-mean in log space: mean=1 in linear space
        mu = -0.5 * sigma_rel ** 2
        return rng.lognormal(mean=mu, sigma=sigma_rel, size=sigma_rel.shape)

    def gaussian_noise_generator(self, z, seed: int = None, max_sigma=25.0):
        """
        Generate additive normal noise for wind components (u, v, w)
        with altitude-dependent standard deviations.

        Parameters:
            z : np.ndarray
                1D array of altitudes [m] at which to compute noise.
            seed : int, optional
                Seed for reproducibility.
            max_sigma : float, optional
                Maximum allowed standard deviation (in wind units, e.g., m/s).

        Returns:
            np.ndarray:
                Array of shape (3, len(z)) with noise for u, v, and w.
        """
        rng = np.random.default_rng(seed)

        # Efficient interpolation using xarray
        sigma_z = self.sigma.interp(z_mc=z, method='linear', kwargs={'fill_value': "extrapolate"})
        sigma_arr = sigma_z.to_array().clip(0, max_sigma).values  # shape: (3, len(z))

        # Additive Gaussian noise (no scaling by √dt)
        return rng.normal(loc=0.0, scale=sigma_arr, size=sigma_arr.shape)

    def _apply_noise(self, wind_vector: np.ndarray, z: np.ndarray, seed: int) -> np.ndarray:
        """
        Apply noise to the wind vector based on the configured noise type.

        Parameters
        ----------
        wind_vector : np.ndarray
            Array of shape (3, num_particles) representing u, v, w wind components.
        z : np.ndarray
            Array of shape (num_particles, ) with particle altitudes [meters].
        seed : float
            Seed or scalar controlling noise realization.

        Returns
        -------
        np.ndarray
            Wind vector with noise applied (shape unchanged).
        """
        if seed == 0 or seed is False:
            return wind_vector

        if self.noise_type == 'lognormal':
            noise = self.lognormal_noise_generator(z, seed=seed, max_sigma=1.5)
            return wind_vector * noise

        elif self.noise_type == 'gaussian':
            noise = self.gaussian_noise_generator(z, seed=seed, max_sigma=25.0)
            return wind_vector + noise
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

    def velocity_vectorized(self, time, state, noise_scale=False):
        """
        Vectorized computation of wind velocity for all particles at given positions and time.

        Parameters:
            time : float
                Time in seconds since self.start_time
            state : ndarray
                Array of shape (num_particles * state_size) — flat.
            noise_scale : float or False
                If non-zero, apply multiplicative lognormal noise with sigma=0.15

        Returns:
            Flattened array of shape (num_particles * state_size)
        """

        # Reshape to (num_particles, state_size)
        x, y, z = np.asarray(state).reshape(self.state_size, -1)

        # Convert time from seconds since start to timestamp
        timestamp = pd.to_datetime(time, origin=self.start_time, unit='s')

        # Prepare interpolation points as xarray Dataset
        coords = xr.Dataset(
            {dim: (('points',), vals)
             for dim, vals in zip(self.interp_dims, [np.repeat(timestamp, x.size), y, x, z, z])
             }
        )

        # Interpolate wind field at all particle positions
        wind = self.wind.interp(coords, method=self.interp_method, kwargs={'fill_value': 0.0})

        # Stack into (num_particles, state_size)
        wind_vector = np.stack([wind.u.data, wind.v.data, wind.w.data])

        # Add lognormal noise if enabled
        wind_vector = self._apply_noise(wind_vector, z, seed=noise_scale)

        # Return as a flat array: shape (num_particles * state_size,...)
        return wind_vector.reshape(-1)

    def intersection_event(self, time, state, target=None, distance_tolerance=1e3):
        """
        Check if any particle trajectory intersects with the target location.

        Parameters:
        - time: float, time in seconds since self.start_time
        - state: flat ndarray, shape (state_size * num_particles),
            representing the current state of the particles
        - target: xarray.Dataset with 'lon', 'lat', 'z' coordinates for the target position
        - distance_tolerance: float [meters].
            Minimum 3D distance at which a particle is considered to intersect

        Returns:
        - float: Event function value. Negative or zero triggers an event in solve_ivp.
        """
        if target is None:
            return distance_tolerance + 1.0

        timestamp = self.start_time + pd.to_timedelta(time, unit='s')
        x, y, z = np.asarray(state).reshape(self.state_size, -1)

        try:
            target_point = target.sel(time=timestamp, method='nearest')
        except Exception as e:
            if self.verbose > 0:
                print(
                    f"Warning: Failed to find nearest target time at {timestamp}. "
                    f"Skipping event. Error: {e}")
            return distance_tolerance + 1.0

        x0, y0 = transformer.transform(target_point.lon.values, target_point.lat.values)
        z0 = target_point.z.values * 1e3

        distance = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
        event_values = np.nanmin(distance) - distance_tolerance

        if self.verbose > 1:
            print(f"Event check at {timestamp}: distance={1e-3 * np.nanmin(distance)} km")

        return event_values

    def advect_particles(self, start_positions, duration=None, end_date=None,
                         ensemble_size=None, target=None, distance_tolerance=None):
        """
        Compute trajectories for a set of particles.

        Parameters:
        - start_positions: list of (lat, lon, z) tuples
        - duration: Length of the simulation in time units (default: None)
        - end_date: End date of the simulation. If given, duration is ignored.
        - ensemble_size: Number of ensemble members (default: None)
        - target: xarray.Dataset or dict-like structure containing target locations with 'lat', 'lon', and 'z' entries.
                  Used to trigger early termination when any particle comes within the specified distance_tolerance.
        - distance_tolerance: float [meters]. Minimum 3D distance (Euclidean) at which a particle is considered to have
                              intersected the target. Used in the event function to stop integration.

        Returns:
        - xarray.Dataset containing particle trajectories with dimensions ('time', 'particle', 'ensemble')
        """
        if end_date is not None:
            print(f"Using end date: {end_date}. Ignoring duration parameter.")
            duration = pd.to_datetime(parser.parse(end_date)) - pd.to_datetime(self.start_time)

        times = generate_eval_time(duration, self.timestep)
        rel_time = self.rel_time.astype(float)
        times = times[(times >= np.min(rel_time)) & (times <= np.max(rel_time))]
        time_span = (times[0], times[-1])

        initial_positions_m = np.array([
            (*transformer.transform(lon, lat), alt)
            for lon, lat, alt in start_positions
        ]).T.reshape(-1)

        if ensemble_size is None:
            ensemble_size = 1

        num_particles, state_size = np.shape(start_positions)[:2]
        relative_tolerance = np.tile([5e-5, 1e-5, 1e-4], num_particles)
        absolute_tolerance = np.tile([10, 10, 5], num_particles)

        def intersection_event(t, y):
            return self.intersection_event(t, y, target=target,
                                           distance_tolerance=distance_tolerance or 500e3)

        intersection_event.terminal = True
        intersection_event.direction = -1.0

        def integrate_ensemble_member(member):
            if self.verbose > 0:
                print(f"Integrating {num_particles} particles for member {member} over {duration}")

            velocity_func = partial(self.velocity_vectorized, noise_scale=member)

            result = solve_ivp(
                velocity_func,
                t_span=time_span,
                y0=initial_positions_m,
                t_eval=None,
                method=self.method,
                events=intersection_event,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
                dense_output=True
            )

            if self.verbose > 0:
                if result.t_events and result.t_events[0].size > 0:
                    event_time = result.t_events[0][0]
                    event_time = self.start_time + pd.Timedelta(event_time, unit='s')
                    print(f"  Member {member}: Intersection happened at {event_time}.")

            # Constrain the result to the valid time range due to early termination
            solver_times = result.t

            t_start, t_stop = solver_times[0], solver_times[-1]
            valid_times = times[(times >= min(t_start, t_stop)) & (times <= max(t_start, t_stop))]

            result = result.sol(valid_times)
            result = result.reshape(state_size, num_particles, -1).transpose(0, 2, 1)

            return member, valid_times, result

        results = Parallel(n_jobs=-1)(
            delayed(integrate_ensemble_member)(member)
            for member in range(ensemble_size)
        )

        min_valid_length = min(r[1].size for r in results)
        valid_times = results[0][1][:min_valid_length]

        trajectories = np.full((state_size, min_valid_length, num_particles, ensemble_size), np.nan)
        for j, _, trajectory in results:
            trajectories[..., :, j] = trajectory[:, :min_valid_length, :]

        trajectories[:2] = transformer.transform(*trajectories[:2], direction='INVERSE')
        valid_times = pd.to_datetime(valid_times, origin=self.start_time, unit='s')

        trajectories = xr.Dataset(
            {
                "lon": (("time", "particle", "ensemble"), trajectories[0]),
                "lat": (("time", "particle", "ensemble"), trajectories[1]),
                "z": (("time", "particle", "ensemble"), 1e-3 * trajectories[2])
            },
            coords={
                "particle": np.arange(num_particles),
                "ensemble": np.arange(ensemble_size),
                "time": valid_times
            }
        )
        trajectories['z'].attrs['standard_name'] = 'geometric height'
        trajectories['z'].attrs['units'] = 'kilometers'

        trajectories.attrs.update({
            "description": "Lagrangian ensemble back-trajectories.",
            "start_time": str(self.start_time),
            "duration": str(duration),
            "initial_longitude": [round(pos[0], 2) for pos in start_positions],
            "initial_latitude": [round(pos[1], 2) for pos in start_positions],
            "initial_altitudes": [round(pos[2], 2) for pos in start_positions],
            "time_step": self.timestep or "60s",
            "time_lag": self.time_lag or "0h",
            "solver_method": self.method,
            "interpolation_method": self.interp_method,
            "noise_type": self.noise_type,
        })

        if end_date is not None:
            trajectories.attrs['end_time'] = str(parser.parse(end_date))

        return trajectories


if __name__ == "__main__":
    # Load simulated 3D wind field (example)
    base_path = "/home/deterministic-nonperiodic/IAP/Experiments/falcon"
    filename = os.path.join(base_path, "UA-ICON_NWP_atm_DOM01_falcon2_80-120_20250219T03-20T01.nc")

    # Dataset containing 'u', 'v', 'w'
    wind_data = xr.open_dataset(filename)

    # Load the Falcon orbit from ESA
    falcon_orbit = read_falcon(os.path.join(base_path, 'Trajectory_2025-02-19/orbgen#12.dat'))
    falcon_orbit = falcon_orbit[falcon_orbit['GAlt'] < 110]

    target_orbit = falcon_orbit.set_index('timestamp').to_xarray()
    target_orbit = target_orbit[['GLon', 'GLat', 'GAlt']].rename(
        {'timestamp': 'time', 'GLon': 'lon', 'GLat': 'lat', 'GAlt': 'z'}
    )
    
    # Define start time with format yyyy-mm-ddTHH:MM:SS
    start_date = "2025-02-20T00:21:16"
    end_date = "2025-02-19T03:40:00"

    # Define initial particle positions (lon, lat, z[meters]):
    lidar_lon = 11.771847  # 11.46
    lidar_lat = 54.116714  # 54.07
    altitudes = [95.9, 96.1, 96.9, 97.1]

    initial_positions = [(lidar_lon, lidar_lat, 1e3 * altitude) for altitude in altitudes]

    # Define the time-step (can be negative for backward trajectory calculation)
    time_step = "20 min"  # Time step with flexible units [min, m, s, sec, h, hours, etc...]

    solver_method = 'RK45'  # options: ['RK23', 'RK45', 'DOP853', 'LSODA']
    interp_method = 'linear'  # options: ['linear', 'nearest', 'cubic', 'quadratic']
    noise_type = 'lognormal'  # options: ['lognormal', 'gaussian', 'ou', None]

    ensemble_size = 1000  # Number of ensemble members

    # Store a file as netCDF
    filename = os.path.join(
        'data',
        f"trajectories_{solver_method}_{interp_method}_{noise_type}_{start_date}--{end_date}_"
        f"lag:{time_lag}_particles:{len(initial_positions)}_members:{ensemble_size}.nc"
    )

    # Initialize the trajectory calculator: Using Runge-Kutta 2nd order method
    if not os.path.exists(filename):

        solver = LagrangianTrajectories(wind_data,
                                        timestep=time_step, start_time=start_date,
                                        integration_method=solver_method,
                                        interpolation_method=interp_method,
                                        noise_type=noise_type, verbose_level=1)

        # Compute trajectories
        trajectories_dataset = solver.advect_particles(initial_positions,
                                                       end_date=end_date,
                                                       ensemble_size=ensemble_size,
                                                       target=target_orbit)

        save_cf_compliant(trajectories_dataset, filename)

    else:
        trajectories_dataset = xr.open_dataset(filename)

    # Visualize trajectories
    fig_name = (f"./figures/trajectories_{solver_method}"
                f"_{interp_method}_{start_date}--{end_date}_lag-{time_lag}")

    # plot map
    visualize_trajectories_percentile_kde(trajectories_dataset,
                                          wind=wind_data,
                                          orbit=falcon_orbit, figure_name=fig_name)

    # plot intersections
    plot_orbit_and_ensemble_3d(trajectories_dataset, falcon_orbit,
                               max_ensemble=trajectories_dataset.ensemble.size,
                               particle_subset=None,
                               horiz_tol_km=50, vert_tol_km=10,
                               figsize=(8, 7), elev=20, azim=-60,
                               figure_name=f"./figures/orbit_vs_ensemble_3d_{end_date}.png")
