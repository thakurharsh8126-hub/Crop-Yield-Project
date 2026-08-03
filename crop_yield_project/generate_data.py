"""
generate_data.py
Generates a synthetic but realistic crop yield dataset and saves it as CSV.
Run this once before running the main app if crop_data.csv does not exist.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 1500

soil_types = ["Sandy", "Clay", "Loamy", "Silty", "Peaty"]
crop_types = ["Wheat", "Rice", "Maize", "Sugarcane", "Cotton"]

soil_yield_factor = {"Sandy": 0.8, "Clay": 0.9, "Loamy": 1.2, "Silty": 1.0, "Peaty": 0.95}
crop_base_yield = {"Wheat": 3.0, "Rice": 4.0, "Maize": 3.5, "Sugarcane": 6.0, "Cotton": 2.0}

rows = []
for _ in range(N):
    rainfall = np.random.uniform(200, 3000)        # mm
    temperature = np.random.uniform(10, 45)         # Celsius
    humidity = np.random.uniform(20, 100)            # %
    soil = np.random.choice(soil_types)
    fertilizer = np.random.uniform(0, 300)           # kg/hectare
    crop = np.random.choice(crop_types)

    base = crop_base_yield[crop]
    soil_factor = soil_yield_factor[soil]

    # Non-linear-ish relationships to make it interesting for ML
    rain_effect = -((rainfall - 1200) ** 2) / 2_000_000 + 1.0
    temp_effect = -((temperature - 25) ** 2) / 400 + 1.0
    humidity_effect = 0.5 + (humidity / 200)
    fert_effect = 0.6 + (fertilizer / 500)

    yield_val = (
        base
        * soil_factor
        * max(rain_effect, 0.2)
        * max(temp_effect, 0.2)
        * humidity_effect
        * fert_effect
    )

    yield_val += np.random.normal(0, 0.3)  # noise
    yield_val = max(yield_val, 0.1)

    rows.append([rainfall, temperature, humidity, soil, fertilizer, crop, round(yield_val, 2)])

df = pd.DataFrame(
    rows,
    columns=[
        "Rainfall_mm",
        "Temperature_C",
        "Humidity_pct",
        "Soil_Type",
        "Fertilizer_kg_per_ha",
        "Crop_Type",
        "Yield_tons_per_ha",
    ],
)

df.to_csv("crop_data.csv", index=False)
print("crop_data.csv generated with", len(df), "rows")
print(df.head())
