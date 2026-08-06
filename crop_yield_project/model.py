

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_PATH = os.path.join(os.path.dirname(__file__), "crop_data.csv")


class CropYieldModel:
    def __init__(self):
        self.model = None
        self.soil_encoder = LabelEncoder()
        self.crop_encoder = LabelEncoder()
        self.feature_cols = [
            "Rainfall_mm",
            "Temperature_C",
            "Humidity_pct",
            "Soil_Type_enc",
            "Fertilizer_kg_per_ha",
            "Crop_Type_enc",
        ]
        self.metrics = {}
        self.df = None
        self.feature_importance = None

    def load_data(self):
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(
                "crop_data.csv not found. Run generate_data.py first."
            )
        self.df = pd.read_csv(DATA_PATH)
        return self.df

    def train(self):
        df = self.load_data().copy()

        df["Soil_Type_enc"] = self.soil_encoder.fit_transform(df["Soil_Type"])
        df["Crop_Type_enc"] = self.crop_encoder.fit_transform(df["Crop_Type"])

        X = df[self.feature_cols]
        y = df["Yield_tons_per_ha"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        self.metrics = {
            "r2": r2_score(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
        }
        self.y_test = y_test
        self.preds = preds

        self.feature_importance = pd.Series(
            self.model.feature_importances_, index=self.feature_cols
        ).sort_values(ascending=False)

        return self.metrics

    def predict(self, rainfall, temperature, humidity, soil_type, fertilizer, crop_type):
        soil_enc = self.soil_encoder.transform([soil_type])[0]
        crop_enc = self.crop_encoder.transform([crop_type])[0]

        X = pd.DataFrame(
            [[rainfall, temperature, humidity, soil_enc, fertilizer, crop_enc]],
            columns=self.feature_cols,
        )
        return self.model.predict(X)[0]

    def get_soil_types(self):
        return sorted(self.soil_encoder.classes_.tolist())

    def get_crop_types(self):
        return sorted(self.crop_encoder.classes_.tolist())


if __name__ == "__main__":
    m = CropYieldModel()
    metrics = m.train()
    print("Model trained.")
    print(f"R2:   {metrics['r2']:.3f}")
    print(f"MAE:  {metrics['mae']:.3f}")
    print(f"RMSE: {metrics['rmse']:.3f}")
    print("\nFeature importance:")
    print(m.feature_importance)

    sample = m.predict(1200, 25, 60, "Loamy", 150, "Wheat")
    print(f"\nSample prediction (Loamy, Wheat): {sample:.2f} tons/ha")
