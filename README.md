# ⚡ MPC Smart Grid Energy Management System

A **Model Predictive Control (MPC)** simulation that intelligently routes power
between **Solar**, **Battery**, and **Grid** using machine-learning predictions.

Built with Python — no special hardware needed.

![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![License MIT](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

---

## 📸 What You'll See

When you run the dashboard, a live simulation window opens with **5 panels**:

| Panel | What it shows |
|-------|---------------|
| **Power Flow** | Animated arrows showing which source is powering the load right now |
| **Time Series** | Solar generation vs load demand curves over time |
| **Source Pie** | Percentage of time spent on Solar / Battery / Grid |
| **Battery Bar** | Current state-of-charge with safe-zone markers |
| **Alert Card** | Plain-English status message (e.g. "Solar active — surplus charging battery") |

---

## 🧠 How It Works

### AI Prediction (Random Forest)
| Model | Input | Output | Accuracy |
|-------|-------|--------|----------|
| **Solar** | Time + Weather (irradiation, temperature) | DC Power (W) | R² ≈ 0.99 |
| **Load** | Time + Voltage | Household consumption (W) | R² ≈ 0.99 |

### MPC Decision Logic
Every second, the controller checks:

```
IF solar ≥ load       →  Use SOLAR, charge battery with surplus
ELIF battery > 20%    →  Use BATTERY to cover the gap
ELSE                  →  Use GRID (also trickle-charges the battery)
```

### Battery Model
| Parameter | Value |
|-----------|-------|
| Capacity | 10 kWh |
| Charge efficiency | 95% |
| Discharge efficiency | 92% |
| Minimum SOC (protected) | 20% |

---

## 🚀 Quick Start (3 Steps)

### Prerequisites
- **Python 3.8+** installed ([download here](https://www.python.org/downloads/))
- **Git** installed ([download here](https://git-scm.com/downloads))

### Step 1 — Clone the repository

```bash
git clone https://github.com/Jebin-05/MPC-Smart-Grid-Energy-Management.git
cd MPC-Smart-Grid-Energy-Management
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> Only 5 lightweight packages: `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`

### Step 3 — Download the datasets

Download these two datasets and place them as shown below:

| Dataset | Download Link |
|---------|--------------|
| **Solar Power Generation** | [Kaggle — Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) |
| **Household Power Consumption** | [UCI — Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) |

Create this folder structure inside the project:

```
Datasets/
├── Solar_power_Generation_Data/
│   ├── Plant_1_Generation_Data.csv
│   └── Plant_1_Weather_Sensor_Data.csv
└── Household_Power_Consumption_Data/
    └── household_power_consumption.txt
```

### Step 4 — Train the AI models (one-time, ~30 seconds)

```bash
python train_models.py
```

You should see:
```
Solar  → MAE: ~168 W | R²: 0.9864
Load   → MAE: ~23 W  | R²: 0.9986
TRAINING COMPLETE
```

### Step 5 — Launch the dashboard

```bash
python dashboard.py
```

A window opens with the live simulation! 🎉

---

## 🎮 Dashboard Controls

| Key | Action |
|-----|--------|
| **Space** | Pause / Resume simulation |
| **↑ Arrow** | Speed up |
| **↓ Arrow** | Slow down |
| **Q** | Quit |

You can also set the initial speed:

```bash
python dashboard.py --speed 120    # 2× default speed
python dashboard.py --speed 30     # Half speed (more detail)
```

---

## 📁 Project Structure

```
MPC-Smart-Grid-Energy-Management/
│
├── train_models.py      ← Trains the AI models (run once)
├── dashboard.py         ← Launches the simulation dashboard
├── requirements.txt     ← Python dependencies (5 packages)
├── README.md            ← You are here
├── .gitignore
│
└── Datasets/            ← You download these (see Step 3)
    ├── Solar_power_Generation_Data/
    │   ├── Plant_1_Generation_Data.csv
    │   └── Plant_1_Weather_Sensor_Data.csv
    └── Household_Power_Consumption_Data/
        └── household_power_consumption.txt
```

**After running `train_models.py`, these files are generated locally:**
- `solar_model.pkl` — Solar prediction model
- `load_model.pkl` — Load prediction model  
- `solar_features.pkl` — Feature list for solar model
- `load_features.pkl` — Feature list for load model

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'xyz'` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: solar_model.pkl` | Run `python train_models.py` first |
| `FileNotFoundError: Datasets/...` | Download datasets — see Step 3 above |
| `No module named 'tkinter'` | Install: `sudo apt install python3-tk` (Linux) or use Anaconda Python |
| Dashboard window doesn't appear | Make sure you're not running via SSH without X11 forwarding |
| Training is slow | Normal — the household dataset is large. Takes about 30 seconds |

---

## 📊 Typical Simulation Output

A 24-hour cycle looks like this:

| Time | Solar | Load | Source | Battery |
|------|-------|------|--------|---------|
| 04:00 | 0 W | 1058 W | Battery | 74% |
| 06:00 | 13267 W | 1133 W | ☀ Solar | 100% |
| 12:00 | 12698 W | 977 W | ☀ Solar | 100% |
| 19:00 | 0 W | 981 W | 🔋 Battery | 95% |
| 23:00 | 0 W | 1060 W | 🔋 Battery | 50% |
| 02:30 | 0 W | 1064 W | ⚡ Grid | 22% |
| 05:30 | 12994 W | 1063 W | ☀ Solar | 100% |

**Pattern:** Solar charges battery all day → Battery powers the night → Grid kicks in only when battery hits 20%

---

## 📜 License

MIT — Free for personal, academic, and commercial use.

---

## 🙋 Author

**Jebin** — [GitHub](https://github.com/Jebin-05)
