#!/usr/bin/env python3
"""
MPC Smart Grid — AI Model Training
===================================
Trains Random Forest models for solar generation and load demand prediction.
Run this ONCE before launching the dashboard.

Usage:
    python train_models.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
SOLAR_GEN = "Datasets/Solar_power_Generation_Data/Plant_1_Generation_Data.csv"
SOLAR_WEATHER = "Datasets/Solar_power_Generation_Data/Plant_1_Weather_Sensor_Data.csv"
LOAD_DATA = "Datasets/Household_Power_Consumption_Data/household_power_consumption.txt"


def _cyclical(values, period):
    """Return (sin, cos) cyclical encoding."""
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SOLAR DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  MPC SMART GRID — MODEL TRAINING")
print("=" * 60)

print("\n[1/4] Loading solar data …")
gen = pd.read_csv(SOLAR_GEN)
weather = pd.read_csv(SOLAR_WEATHER)

gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], format="%d-%m-%Y %H:%M")
weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"])

solar_df = pd.merge(gen, weather, on=["DATE_TIME", "PLANT_ID"], how="inner")

solar_df["Hour"] = solar_df["DATE_TIME"].dt.hour
solar_df["Minute"] = solar_df["DATE_TIME"].dt.minute
solar_df["DayOfWeek"] = solar_df["DATE_TIME"].dt.dayofweek
solar_df["Month"] = solar_df["DATE_TIME"].dt.month
solar_df["IsWeekend"] = (solar_df["DayOfWeek"] >= 5).astype(int)
solar_df["Hour_sin"], solar_df["Hour_cos"] = _cyclical(solar_df["Hour"], 24)
solar_df["Month_sin"], solar_df["Month_cos"] = _cyclical(solar_df["Month"], 12)

SOLAR_FEATURES = [
    "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION",
    "Hour", "Minute", "DayOfWeek", "Month", "IsWeekend",
    "Hour_sin", "Hour_cos", "Month_sin", "Month_cos",
]
solar_df = solar_df.dropna(subset=SOLAR_FEATURES + ["DC_POWER"])
X_solar = solar_df[SOLAR_FEATURES].values
y_solar = solar_df["DC_POWER"].values
print(f"   {len(solar_df)} records, {len(SOLAR_FEATURES)} features")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/4] Loading household load data …")
load_df = pd.read_csv(LOAD_DATA, sep=";", low_memory=False)
load_df["DateTime"] = pd.to_datetime(
    load_df["Date"] + " " + load_df["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
)
for col in ("Global_active_power", "Voltage", "Global_intensity"):
    load_df[col] = pd.to_numeric(load_df[col], errors="coerce")

load_df["Hour"] = load_df["DateTime"].dt.hour
load_df["Minute"] = load_df["DateTime"].dt.minute
load_df["DayOfWeek"] = load_df["DateTime"].dt.dayofweek
load_df["Month"] = load_df["DateTime"].dt.month
load_df["IsWeekend"] = (load_df["DayOfWeek"] >= 5).astype(int)
load_df["Hour_sin"], load_df["Hour_cos"] = _cyclical(load_df["Hour"], 24)
load_df["Month_sin"], load_df["Month_cos"] = _cyclical(load_df["Month"], 12)
load_df["Power_W"] = load_df["Global_active_power"] * 1000  # kW → W

LOAD_FEATURES = [
    "Voltage", "Global_intensity",
    "Hour", "Minute", "DayOfWeek", "Month", "IsWeekend",
    "Hour_sin", "Hour_cos", "Month_sin", "Month_cos",
]
load_df = load_df.dropna(subset=LOAD_FEATURES + ["Power_W"])
load_df = load_df.sample(frac=0.1, random_state=42)  # 10 % for speed
X_load = load_df[LOAD_FEATURES].values
y_load = load_df["Power_W"].values
print(f"   {len(load_df)} records, {len(LOAD_FEATURES)} features")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN RANDOM FOREST MODELS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/4] Training Random Forest models …")

Xtr, Xte, ytr, yte = train_test_split(X_solar, y_solar, test_size=0.2, random_state=42)
solar_model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
solar_model.fit(Xtr, ytr)
solar_mae = mean_absolute_error(yte, solar_model.predict(Xte))
solar_r2 = r2_score(yte, solar_model.predict(Xte))
print(f"   Solar  → MAE: {solar_mae:.1f} W | R²: {solar_r2:.4f}")

Xtr, Xte, ytr, yte = train_test_split(X_load, y_load, test_size=0.2, random_state=42)
load_model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
load_model.fit(Xtr, ytr)
load_mae = mean_absolute_error(yte, load_model.predict(Xte))
load_r2 = r2_score(yte, load_model.predict(Xte))
print(f"   Load   → MAE: {load_mae:.1f} W | R²: {load_r2:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. SAVE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/4] Saving models …")
joblib.dump(solar_model, "solar_model.pkl")
joblib.dump(load_model, "load_model.pkl")
joblib.dump(SOLAR_FEATURES, "solar_features.pkl")
joblib.dump(LOAD_FEATURES, "load_features.pkl")

print("   ✓ solar_model.pkl")
print("   ✓ load_model.pkl")
print("   ✓ solar_features.pkl")
print("   ✓ load_features.pkl")

print("\n" + "=" * 60)
print("  TRAINING COMPLETE — You can now run: python dashboard.py")
print("=" * 60 + "\n")
