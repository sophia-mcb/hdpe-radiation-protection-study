import pandas as pd
import numpy as np
from pathlib import Path

input_path = Path("/mnt/data/sun_geiger_data.csv")
df = pd.read_csv(input_path)

# Build UTC timestamps from the GPS fields.
df["datetime_utc"] = pd.to_datetime(
    dict(
        year=df["GPS Year"],
        month=df["GPS Month"],
        day=df["GPS Day"],
        hour=df["GPS Hour"],
        minute=df["GPS Minute"],
        second=df["GPS Second"],
    ),
    utc=True,
)

# Self-contained solar-zenith calculation used 
def solar_zenith_noaa(dt, lat_deg, lon_deg):
    n = dt.dayofyear
    hour = dt.hour + dt.minute / 60 + dt.second / 3600
    gamma = 2 * np.pi / 365 * (n - 1 + (hour - 12) / 24)

    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )

    true_solar_time = hour * 60 + eqtime + 4 * lon_deg
    hour_angle = np.deg2rad(true_solar_time / 4 - 180)
    lat = np.deg2rad(lat_deg)

    cos_zenith = (
        np.sin(lat) * np.sin(decl)
        + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    )
    return np.rad2deg(np.arccos(np.clip(cos_zenith, -1, 1)))

df["solar_zenith_deg"] = [
    solar_zenith_noaa(t, lat, lon)
    for t, lat, lon in zip(
        df["datetime_utc"], df["GPS Latitude"], df["GPS Longitude"]
    )
]

I0 = 1361.0  # W/m^2
offsets_m = {"left": -0.13, "middle": 0.0, "right": 0.13}
k_values = [0.0001, 0.00015, 0.0002]

altitude_m = np.maximum(df["GPS Altitude"].to_numpy(dtype=float), 1e-9)
zenith_rad = np.deg2rad(df["solar_zenith_deg"].to_numpy(dtype=float))
elevation_rad = np.deg2rad(90.0 - df["solar_zenith_deg"].to_numpy(dtype=float))

summary_rows = []

for k in k_values:
    irradiance = {}
    for sensor, x_offset in offsets_m.items():
        theta = zenith_rad + np.arctan2(
            x_offset,
            altitude_m * np.tan(elevation_rad)
        )
        irradiance[sensor] = (
            I0
            * np.cos(theta)
            * np.exp(-k * altitude_m / np.cos(zenith_rad))
        )

    # Mean of the two end sensors versus the middle sensor.
    end_mean = (irradiance["left"] + irradiance["right"]) / 2
    relative_end_vs_middle = (end_mean - irradiance["middle"]) / irradiance["middle"] * 100

    # Also retain each end separately because the two effects largely cancel
    # when averaged symmetrically.
    left_rel = (irradiance["left"] - irradiance["middle"]) / irradiance["middle"] * 100
    right_rel = (irradiance["right"] - irradiance["middle"]) / irradiance["middle"] * 100

    for label, mask in [
        ("all 107 bins", np.ones(len(df), dtype=bool)),
        ("altitude > 10 km", altitude_m > 10000),
        ("altitude > 20 km", altitude_m > 20000),
    ]:
        summary_rows.append({
            "k_m^-1": k,
            "subset": label,
            "end_mean_vs_middle_mean_percent": np.mean(relative_end_vs_middle[mask]),
            "mean_absolute_end_mean_difference_percent": np.mean(np.abs(relative_end_vs_middle[mask])),
            "left_vs_middle_mean_percent": np.mean(left_rel[mask]),
            "right_vs_middle_mean_percent": np.mean(right_rel[mask]),
            "maximum_absolute_end_difference_percent": np.max(
                np.maximum(np.abs(left_rel[mask]), np.abs(right_rel[mask]))
            ),
        })

# Save the reconstructed per-bin model output and summary.
result = df[[
    "datetime_utc", "GPS Latitude", "GPS Longitude", "GPS Altitude",
    "Sun Azimuth", "solar_zenith_deg", "Pitch", "Roll", "Yaw",
    "SpcS POD1-01", "Cntrl POD1-02", "HDPE POD1-03"
]].copy()

# Use the middle attenuation value for the detailed output.
k = 0.00015
for sensor, x_offset in offsets_m.items():
    theta = zenith_rad + np.arctan2(
        x_offset, altitude_m * np.tan(elevation_rad)
    )
    result[f"irradiance_{sensor}_W_m2"] = (
        I0 * np.cos(theta) * np.exp(-k * altitude_m / np.cos(zenith_rad))
    )

result["end_mean_vs_middle_percent"] = (
    (
        (result["irradiance_left_W_m2"] + result["irradiance_right_W_m2"]) / 2
        - result["irradiance_middle_W_m2"]
    )
    / result["irradiance_middle_W_m2"]
    * 100
)

summary = pd.DataFrame(summary_rows)

result_path = Path("/mnt/data/reconstructed_solar_model_by_bin.csv")
summary_path = Path("/mnt/data/reconstructed_solar_model_summary.csv")
script_path = Path("/mnt/data/reconstruct_solar_model.py")

result.to_csv(result_path, index=False)
summary.to_csv(summary_path, index=False)

script = r'''#!/usr/bin/env python3
"""
Solar-exposure model described in Section 2.6.2.

"""
'''
script_path.write_text(script)

print("Rows:", len(df))
print("Maximum GPS altitude:", df["GPS Altitude"].max(), "m")
print()
print(summary.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
print()
print("Files created:")
print(result_path)
print(summary_path)
print(script_path)
