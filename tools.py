import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d
from pyproj import Geod


def save_cf_compliant(ds: xr.Dataset, path: str):
    """Save with CF-compliant time encoding compatible with CDO."""
    time_origin = pd.to_datetime(ds.time.values[0])
    origin_str = time_origin.strftime("%Y-%m-%d %H:%M:%S")

    encoding = {var: {"zlib": True} for var in ds.data_vars}
    encoding["time"] = {
        "units": f"seconds since {origin_str}",
        "calendar": "proleptic_gregorian",
        "dtype": "float64"
    }

    ds.to_netcdf(path, encoding=encoding)


def sigma_component(component="u"):
    """
    Create a function to vertically interpolate the standard deviation of wind components
    :param component: string, one of ['u', 'v', 'w']
    :return: interpolation functions for the specified wind component
    """

    # Define altitude and sigma values for wind components
    altitude = 1e3 * np.array([80.496086, 81.70819, 82.933266, 84.171394, 85.42264, 86.68708,
                               87.96479, 89.255844, 90.56032, 91.87829, 93.20984, 94.55504,
                               95.91399, 97.28675, 98.673416, 100.07407])

    sigma = {
        "u": np.array([21.072638, 19.778282, 18.733944, 17.160027, 15.1502495, 13.419383,
                       13.337099, 15.494116, 18.56262, 22.313557, 26.949926, 31.411144,
                       35.637962, 38.088654, 39.60116, 40.94975]),
        "v": np.array([19.0583, 19.531872, 19.139727, 17.517744, 15.989676, 16.746874,
                       19.3816, 22.937468, 27.505886, 31.952532, 35.201454, 37.258953,
                       39.073856, 40.34578, 39.15983, 37.215282]),
        "w": np.array([0.28650776, 0.29934645, 0.3305828, 0.3605683, 0.37941748, 0.3868968,
                       0.39157304, 0.39810547, 0.41244808, 0.44748694, 0.4843474, 0.511466,
                       0.5366538, 0.5540057, 0.555168, 0.52239084])
    }
    # create interpolation functions
    sigma_interp = interp1d(
        altitude, sigma[component], kind='linear',
        fill_value=(sigma[component][0], sigma[component][-1]), bounds_error=False
    )

    return sigma_interp


def sigma_components(z_coord: str = 'z_mc'):
    """
    Create a dataset with vertical profiles of standard deviation for u, v, and w wind components.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions:
            - 'z_mc': altitude in meters
        Variables:
            - 'u', 'v', 'w': standard deviations (m/s)
    """
    # Altitudes in meters
    altitude = 1e3 * np.array([
        80.496086, 81.70819, 82.933266, 84.171394, 85.42264, 86.68708,
        87.96479, 89.255844, 90.56032, 91.87829, 93.20984, 94.55504,
        95.91399, 97.28675, 98.673416, 100.07407
    ])

    sigma_data = {
        "u": np.array([21.072638, 19.778282, 18.733944, 17.160027, 15.1502495, 13.419383,
                       13.337099, 15.494116, 18.56262, 22.313557, 26.949926, 31.411144,
                       35.637962, 38.088654, 39.60116, 40.94975]),
        "v": np.array([19.0583, 19.531872, 19.139727, 17.517744, 15.989676, 16.746874,
                       19.3816, 22.937468, 27.505886, 31.952532, 35.201454, 37.258953,
                       39.073856, 40.34578, 39.15983, 37.215282]),
        "w": np.array([0.28650776, 0.29934645, 0.3305828, 0.3605683, 0.37941748, 0.3868968,
                       0.39157304, 0.39810547, 0.41244808, 0.44748694, 0.4843474, 0.511466,
                       0.5366538, 0.5540057, 0.555168, 0.52239084])
    }

    sigma_data = xr.Dataset({
        comp: xr.DataArray(data, dims=[z_coord], coords={z_coord: altitude})
        for comp, data in sigma_data.items()
    })

    return sigma_data


def generate_ou_noise_field(z_vals, times, tau=600, sigma_fn=sigma_component, seed=None):
    """
    Generate a temporally correlated OU noise field on a (time, altitude) grid.

    Parameters:
        z_vals : 1D np.ndarray
            Altitudes (in meters).
        times : 1D np.ndarray
            Time values (in seconds).
        tau : float
            Relaxation time-scale (in seconds).
        sigma_fn : callable
            Function that returns standard deviation given altitude.
        seed : int
            RNG seed.

    Returns:
        xr.Dataset with variables 'u_noise', 'v_noise', 'w_noise' on dimensions (time, z_mc)
    """
    rng = np.random.default_rng(seed)
    dt = np.diff(times).min()
    n_time = len(times)
    n_z = len(z_vals)

    # Shape: (3, n_time, n_z)
    noise = np.zeros((3, n_time, n_z))

    if sigma_fn is None:
        raise ValueError("sigma_fn must be provided.")

    sigma_vals = np.stack([sigma_fn(c)(z_vals) for c in ['u', 'v', 'w']])  # shape: (3, n_z)

    # Initialize with zero
    for i in range(1, n_time):
        decay = np.exp(-dt / tau)
        noise[:, i] = decay * noise[:, i - 1] + \
                      np.sqrt(1 - decay ** 2) * rng.normal(loc=0, scale=sigma_vals, size=(3, n_z))

    # Package into xarray
    return xr.Dataset({
        'u': (['time', 'z_mc'], noise[0]),
        'v': (['time', 'z_mc'], noise[1]),
        'w': (['time', 'z_mc'], noise[2]),
    }, coords={'time': times, 'z_mc': z_vals})


def generate_mean_wind(wind: xr.Dataset) -> xr.DataArray:
    """
    Compute the vertical mean wind speed profile, weighted by cosine(latitude),
    and averaged over time and horizontal space.

    Parameters
    ----------
    wind : xarray.Dataset
        Dataset with variables 'u', 'v', 'w' and coordinates including 'lat', 'lon', 'time', 'z_mc'.

    Returns
    -------
    xr.DataArray
        1D array of mean wind speed vs. vertical level (z_coord).
    """
    print("Generating mean wind speed profile for noise scaling...")

    # Efficient chunking
    wind = wind.chunk({dim: "auto" if dim not in ['lon', 'lat'] else -1 for dim in wind.dims})
    wind = wind.sel(lon=slice(-12, 12), lat=slice(45, 65))

    # Compute magnitude of wind vector: √(u² + v² + w²)
    wind_speed = np.sqrt(wind.u ** 2 + wind.v ** 2 + wind.w ** 2)

    # Weight by cosine latitude (broadcast safely)
    weights = np.cos(np.deg2rad(wind.lat))
    weighted_speed = wind_speed.weighted(weights)

    # Mean over time and horizontal dims, preserving the vertical profile
    mean_profile = weighted_speed.mean(dim=['time', 'lat', 'lon'])

    return mean_profile.compute()


def read_falcon(filename):
    orbit_df = pd.read_csv(
        filename,
        comment="#",  # skip all header lines starting with #
        sep='\s+',  # auto-split by whitespace
        header=None,  # no header in the data row,
        names=[
            "timestamp", "dt_min", "IFA", "F10.7", "FB10.7", "Ap", "IDW", "Rho",
            "Vn", "Ve", "MMWT", "Tloc", "Texo", "LST", "GLat", "GLon", "GAlt",
            "Va", "gamma", "gload", "qdot", "SRat", "TRat", "KnInf", "MaInf",
            "CD", "CD_CD0", "Orb", "ULat", "dS", "Hpe", "Hap", "H", "Torb",
            "mjd1950"
        ]
    )

    orbit_df["timestamp"] = pd.to_datetime(orbit_df["timestamp"])
    orbit_df = orbit_df.drop_duplicates(subset="timestamp")

    return orbit_df


# Helper function to convert duration or timestep to seconds
def convert_to_seconds(value, unit='s'):
    if isinstance(value, (int, float)):
        return pd.Timedelta(value, unit=unit).total_seconds()
    elif isinstance(value, str):
        return pd.Timedelta(value).total_seconds()
    elif isinstance(value, pd.Timedelta):
        return value.total_seconds()
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


def compute_intersections(
        trajectories: xr.Dataset,
        orbit_df: pd.DataFrame,
        horiz_tol_km=50,
        vert_tol_km=5,
        time_tol_m=30
):
    """
    Find where particle trajectories intersect with a re-entry trajectory in space and time.

    Parameters:
        trajectories : xarray.Dataset
            Must include 'lon', 'lat', and 'z' with dims (time, particle, ensemble).
        orbit_df : pd.DataFrame
            Must include ['timestamp', 'GLon', 'GLat', 'GAlt'].
        horiz_tol_km : float
            Horizontal tolerance for intersection [km].
        vert_tol_km : float
            Vertical tolerance for intersection [km].
        time_tol_m : float
            Time tolerance in minutes for considering same-time intersections.

    Returns:
        intersect_mask : xarray.DataArray (bool)
            Mask with shape (time, particle, ensemble) showing intersection flags.
        intersect_indices : np.ndarray
            Indices of shape (N, 3) for (time, particle, ensemble) of intersecting points.
    """

    geode = Geod(ellps="WGS84")

    # Flatten coordinates and time
    traj_time = trajectories.time.values
    flat_time = np.repeat(traj_time[:, None, None],
                          repeats=trajectories.particle.size * trajectories.ensemble.size,
                          axis=1).reshape(-1)

    flat_coords = pd.DataFrame({
        'timestamp': pd.to_datetime(flat_time),
        'lon': trajectories.lon.values.ravel(),
        'lat': trajectories.lat.values.ravel(),
        'z': trajectories.z.values.ravel()
    })

    # Align orbit in time
    # Track original index before sorting
    flat_coords["orig_index"] = flat_coords.index
    flat_coords_sorted = flat_coords.sort_values("timestamp").reset_index(drop=True)

    # Merge with orbit
    orbit_df_sorted = orbit_df.sort_values('timestamp')
    aligned = pd.merge_asof(flat_coords_sorted, orbit_df_sorted,
                            on='timestamp', tolerance=pd.Timedelta(minutes=time_tol_m),
                            direction='nearest')

    # Drop unmatched rows (e.g. where orbit was NaN)
    aligned = aligned.dropna(subset=["GLon", "GLat", "GAlt"])

    # Compute distances
    _, _, dist_m = geode.inv(aligned["lon"].values, aligned["lat"].values,
                             aligned["GLon"].values, aligned["GLat"].values)
    horiz_km = dist_m / 1000.0
    vert_km = np.abs(aligned["z"].values - aligned["GAlt"].values)

    intersect = (horiz_km < horiz_tol_km) & (vert_km < vert_tol_km)

    # Use original indices to construct the mask
    intersect_flat_indices = aligned.loc[intersect, "orig_index"].to_numpy()

    # Build mask
    shape = trajectories.lon.shape
    mask = np.zeros(np.prod(shape), dtype=bool)
    mask[intersect_flat_indices] = True
    intersect_mask = mask.reshape(shape)

    # Final output
    intersect_indices = np.argwhere(intersect_mask)
    intersect_mask = xr.DataArray(intersect_mask,
                                  dims=trajectories.lon.dims,
                                  coords=trajectories.lon.coords)

    return intersect_mask, intersect_indices
