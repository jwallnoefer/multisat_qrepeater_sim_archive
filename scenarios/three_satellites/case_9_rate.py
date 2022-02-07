import os, sys; sys.path.insert(0, os.path.abspath("."))
import libs.matrix as mat
from libs.aux_functions import binary_entropy
from consts import AVERAGE_EARTH_RADIUS as R_E
import numpy as np
import pandas as pd

M = 5.972e24  # earth mass
G = 6.67408e-11  # gravitational constant


def raw_errors_from_data_frame(data_frame):
    times = data_frame["time"]
    raw_rate = len(times) / times.iloc[-1]

    states = data_frame["state"]

    z0z0 = mat.tensor(mat.z0, mat.z0)
    z1z1 = mat.tensor(mat.z1, mat.z1)
    correlations_z = np.real_if_close([(mat.H(z0z0) @ state @ z0z0)[0, 0] + (mat.H(z1z1) @ state @ z1z1)[0, 0] for state in states])
    correlations_z[correlations_z > 1] = 1
    e_z = 1 - np.mean(correlations_z)

    x0x0 = mat.tensor(mat.x0, mat.x0)
    x1x1 = mat.tensor(mat.x1, mat.x1)
    correlations_x = np.real_if_close([(mat.H(x0x0) @ state @ x0x0)[0, 0] + (mat.H(x1x1) @ state @ x1x1)[0, 0] for state in states])
    correlations_x[correlations_x > 1] = 1
    e_x = 1 - np.mean(correlations_x)

    return raw_rate, e_z, e_x


def effective_rate_from_raw_errors(data_frame, orbital_period, err_corr_ineff=1):
    integrand_x = data_frame["raw_rate"] * data_frame["e_x"]
    integrand_z = data_frame["raw_rate"] * data_frame["e_z"]
    normalization = np.trapz(y=data_frame["raw_rate"], x=data_frame.index)
    effective_e_x = np.trapz(y=integrand_x, x=data_frame.index) / normalization
    effective_e_z = np.trapz(y=integrand_z, x=data_frame.index) / normalization
    effective_yield = normalization / orbital_period
    return effective_yield * (1 - binary_entropy(effective_e_x) - err_corr_ineff * binary_entropy(effective_e_z))


def rate_from_data_series(data_series, orbital_height, ground_distance, err_corr_ineff=1):
    offsets = np.array(data_series.index)
    angles = offsets * ground_distance / R_E
    orbital_period = 2 * np.pi * np.sqrt((R_E + orbital_height)**3 / (M * G))
    times = angles / (2 * np.pi) * orbital_period
    data = [list(raw_errors_from_data_frame(df)) for df in data_series]
    new_df = pd.DataFrame(data, index=times, columns=["raw_rate", "e_z", "e_x"])
    return effective_rate_from_raw_errors(data_frame=new_df, orbital_period=orbital_period, err_corr_ineff=err_corr_ineff)


def optimize_rate_from_data_series(data_series, orbital_height, ground_distance, err_corr_ineff=1):
    offsets = np.array(data_series.index)
    angles = offsets * ground_distance / R_E
    orbital_period = 2 * np.pi * np.sqrt((R_E + orbital_height)**3 / (M * G))
    times = angles / (2 * np.pi) * orbital_period
    data = [list(raw_errors_from_data_frame(df)) for df in data_series]
    new_df = pd.DataFrame(data, index=times, columns=["raw_rate", "e_z", "e_x"])
    shortened_dfs = [new_df] + [new_df.iloc[i:-i] for i in range(1, len(new_df) // 2)]
    effective_rates = [effective_rate_from_raw_errors(data_frame=df,
                                                      orbital_period=orbital_period,
                                                      err_corr_ineff=err_corr_ineff)
                       for df in shortened_dfs]
    return np.max(effective_rates)


# first do the twolink cases
result_path = os.path.join("results", "three_satellites", "twolink_downlink", "satellite_path")
orbital_height = 400e3
ground_distance = 4400e3
configurations = [-1, 0, 1, 2]
paths = [os.path.join(result_path, f"{configuration}_configuration") for configuration in configurations]
output = []
# for path in paths:
#     print("Now processing: ", path)
#     data_series = pd.read_pickle(os.path.join(path, "raw_data.bz2"))
#     output += [rate_from_data_series(data_series, orbital_height=orbital_height, ground_distance=ground_distance)]
# res = pd.Series(data=output, index=configurations)
# res.to_csv(os.path.join(result_path, "effective_rate.csv"))
for path in paths:
    print("Now processing: ", path)
    data_series = pd.read_pickle(os.path.join(path, "raw_data.bz2"))
    output += [optimize_rate_from_data_series(data_series, orbital_height=orbital_height, ground_distance=ground_distance)]
res = pd.Series(data=output, index=configurations)
res.to_csv(os.path.join(result_path, "optimized_effective_rate.csv"))

# then same for fourlink case
result_path = os.path.join("results", "three_satellites", "fourlink", "satellite_path")
orbital_height = 400e3
ground_distance = 4400e3
configurations = [-1, 0, 1, 2]
paths = [os.path.join(result_path, f"{configuration}_configuration") for configuration in configurations]
output = []
# for path in paths:
#     print("Now processing: ", path)
#     data_series = pd.read_pickle(os.path.join(path, "raw_data.bz2"))
#     output += [rate_from_data_series(data_series, orbital_height=orbital_height, ground_distance=ground_distance)]
# res = pd.Series(data=output, index=configurations)
# res.to_csv(os.path.join(result_path, "effective_rate.csv"))
for path in paths:
    print("Now processing: ", path)
    data_series = pd.read_pickle(os.path.join(path, "raw_data.bz2"))
    output += [optimize_rate_from_data_series(data_series, orbital_height=orbital_height, ground_distance=ground_distance)]
res = pd.Series(data=output, index=configurations)
res.to_csv(os.path.join(result_path, "optimized_effective_rate.csv"))
