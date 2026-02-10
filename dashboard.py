#!/usr/bin/env python3
"""
MPC Smart Grid — Real-Time Simulation Dashboard
=================================================
A clean, single-file dashboard that visualises Model Predictive Control
for a Solar + Battery + Grid energy system.

Usage:
    python dashboard.py            # run simulation (default)
    python dashboard.py --speed 120  # 2× faster

Prerequisites:
    pip install -r requirements.txt
    python train_models.py          # train the ML models first
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import warnings
from datetime import datetime, timedelta
from collections import deque

import matplotlib
# Try TkAgg first, fall back to any available backend
for _backend in ("TkAgg", "Qt5Agg", "GTK3Agg", "Agg"):
    try:
        matplotlib.use(_backend)
        break
    except Exception:
        continue
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (dark theme)
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "solar":    "#FF9500",
    "battery":  "#34C759",
    "grid":     "#007AFF",
    "load":     "#FF3B30",
    "bg":       "#0F0F0F",
    "panel":    "#1C1C1E",
    "text":     "#FFFFFF",
    "dim":      "#8E8E93",
    "line":     "#2C2C2E",
    "ok":       "#30D158",
    "warn":     "#FFD60A",
    "danger":   "#FF453A",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  AI  ENGINE  —  loads Random-Forest models trained by train_models.py
# ═══════════════════════════════════════════════════════════════════════════════
class AIEngine:
    """Load trained ML models and make predictions."""

    def __init__(self):
        self.solar_model = None
        self.load_model = None
        self.solar_features: list = []
        self.load_features: list = []

    def load(self) -> bool:
        ok = True
        for name, attr, feat_attr, feat_file in [
            ("solar_model.pkl", "solar_model", "solar_features", "solar_features.pkl"),
            ("load_model.pkl",  "load_model",  "load_features",  "load_features.pkl"),
        ]:
            if not os.path.exists(name):
                print(f"  ✗ {name} not found — run train_models.py first")
                ok = False
                continue
            setattr(self, attr, joblib.load(name))
            if os.path.exists(feat_file):
                setattr(self, feat_attr, joblib.load(feat_file))
            print(f"  ✓ {name} loaded")
        return ok

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _time_features(ts: datetime) -> dict:
        h, m = ts.hour, ts.minute
        dow = ts.weekday()
        mon = ts.month
        return {
            "Hour": h, "Minute": m, "DayOfWeek": dow, "Month": mon,
            "IsWeekend": int(dow >= 5),
            "Hour_sin": np.sin(2 * np.pi * h / 24),
            "Hour_cos": np.cos(2 * np.pi * h / 24),
            "Month_sin": np.sin(2 * np.pi * mon / 12),
            "Month_cos": np.cos(2 * np.pi * mon / 12),
        }

    def predict_solar(self, ts: datetime, irr: float, amb_t: float, mod_t: float) -> float:
        if self.solar_model is None:
            return 0.0
        row = {**self._time_features(ts),
               "IRRADIATION": irr,
               "AMBIENT_TEMPERATURE": amb_t,
               "MODULE_TEMPERATURE": mod_t}
        df = pd.DataFrame([row])[self.solar_features]
        return float(max(0, self.solar_model.predict(df)[0]))

    def predict_load(self, ts: datetime, voltage: float = 230.0) -> float:
        if self.load_model is None:
            return 0.0
        row = {**self._time_features(ts), "Voltage": voltage, "Global_intensity": voltage / 50}
        df = pd.DataFrame([row])[self.load_features]
        return float(max(0, self.load_model.predict(df)[0]))


# ═══════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT SIMULATOR  —  realistic solar / weather / load curves
# ═══════════════════════════════════════════════════════════════════════════════
class EnvironmentSim:
    """Generate realistic sensor values for the simulation."""

    @staticmethod
    def irradiation(ts: datetime) -> float:
        h = ts.hour + ts.minute / 60
        if h < 5.5 or h > 18.5:
            return 0.0
        return max(0, 1000 * np.exp(-((h - 12) ** 2) / 8) * np.random.uniform(0.75, 1.0))

    @staticmethod
    def ambient_temp(ts: datetime) -> float:
        h = ts.hour + ts.minute / 60
        return 22 + 8 * np.sin(np.pi * (h - 6) / 12) + np.random.uniform(-1, 1)

    @staticmethod
    def module_temp(amb: float, irr: float) -> float:
        return amb + irr * 0.02 + np.random.uniform(-0.5, 0.5)

    @staticmethod
    def voltage() -> float:
        return np.random.normal(230, 3)


# ═══════════════════════════════════════════════════════════════════════════════
#  MPC  CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════
class MPCController:
    """Simple but correct Model Predictive Control logic."""

    def __init__(self, capacity_wh=10_000, initial_soc=80.0):
        self.capacity = capacity_wh          # Wh
        self.soc = initial_soc               # %
        self.min_soc = 20.0
        self.max_soc = 100.0
        self.charge_eff = 0.95
        self.discharge_eff = 0.92

    def decide(self, solar_w: float, load_w: float):
        """Return (source, reason) where source ∈ {Solar, Battery, Grid}."""
        if solar_w >= load_w:
            return "Solar", f"Solar ({solar_w:.0f} W) covers load ({load_w:.0f} W)"
        if self.soc > self.min_soc:
            return "Battery", f"Battery at {self.soc:.0f}% bridging {load_w - solar_w:.0f} W gap"
        return "Grid", f"Battery low ({self.soc:.0f}%), grid needed"

    def step(self, solar_w: float, load_w: float, source: str, dt_minutes: float):
        """Update battery SOC for one time-step."""
        dt_h = dt_minutes / 60
        if source == "Solar" and solar_w > load_w:
            energy = (solar_w - load_w) * dt_h * self.charge_eff
            self.soc = min(self.max_soc, self.soc + energy / self.capacity * 100)
        elif source == "Battery":
            energy = (load_w - solar_w) * dt_h / self.discharge_eff
            self.soc = max(0, self.soc - energy / self.capacity * 100)
        elif source == "Grid" and self.soc < 50:
            # Trickle-charge from grid when battery is low
            energy = 1500 * dt_h * self.charge_eff
            self.soc = min(self.max_soc, self.soc + energy / self.capacity * 100)


# ═══════════════════════════════════════════════════════════════════════════════
#  ALERT  GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
def generate_alert(source, solar, load, soc):
    """Return a single (type, title, detail) alert tuple."""
    if source == "Solar":
        return ("ok",
                "SOLAR ACTIVE — Clean energy powering load",
                f"Solar {solar:.0f} W  |  Load {load:.0f} W  |  Surplus charging battery")
    if source == "Battery":
        return ("warn",
                "BATTERY ACTIVE — Stored energy in use",
                f"Solar {solar:.0f} W < Load {load:.0f} W  |  Battery {soc:.0f}%")
    return ("danger",
            "GRID ACTIVE — External power required",
            f"Solar {solar:.0f} W  |  Load {load:.0f} W  |  Battery {soc:.0f}% (low)")


# ═══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
class Dashboard:
    """Matplotlib-based real-time control dashboard."""

    HISTORY = 120  # data points to keep

    def __init__(self, speed: int = 60):
        self.ai = AIEngine()
        self.env = EnvironmentSim()
        self.mpc = MPCController()
        self.speed = speed  # simulated minutes per real second

        self.sim_time = datetime(2020, 5, 15, 4, 0)  # start before sunrise
        self.paused = False

        # history buffers
        self.t_hist = deque(maxlen=self.HISTORY)
        self.solar_hist = deque(maxlen=self.HISTORY)
        self.load_hist = deque(maxlen=self.HISTORY)
        self.soc_hist = deque(maxlen=self.HISTORY)
        self.src_hist = deque(maxlen=self.HISTORY)

        # stats
        self.solar_kwh = 0.0
        self.grid_kwh = 0.0

    # ── initialise ───────────────────────────────────────────────────────────
    def init(self):
        print("\n" + "=" * 55)
        print("  MPC SMART GRID — SIMULATION DASHBOARD")
        print("=" * 55)
        print("\n  Loading AI models …")
        if not self.ai.load():
            print("\n  ⚠  Models not found. Run:  python train_models.py\n")
            sys.exit(1)
        print("\n  Building UI …")
        self._build_ui()
        print(f"  Simulation speed: {self.speed} min / sec")
        print("\n  Controls:  SPACE = pause  |  ↑↓ = speed  |  Q = quit")
        print("=" * 55 + "\n")

    # ── UI layout ────────────────────────────────────────────────────────────
    def _build_ui(self):
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(17, 9.5), facecolor=C["bg"])
        self.fig.canvas.manager.set_window_title("MPC Smart Grid Dashboard")

        gs = GridSpec(3, 4, figure=self.fig,
                      hspace=0.42, wspace=0.35,
                      left=0.05, right=0.97, top=0.93, bottom=0.06)

        # Row 0 — Power flow (full width)
        self.ax_flow = self.fig.add_subplot(gs[0, :])

        # Row 1 — Time-series (3 cols) + Source pie (1 col)
        self.ax_ts = self.fig.add_subplot(gs[1, 0:3])
        self.ax_pie = self.fig.add_subplot(gs[1, 3])

        # Row 2 — Battery bar (2 cols) + Alert card (2 cols)
        self.ax_bat = self.fig.add_subplot(gs[2, 0:2])
        self.ax_alert = self.fig.add_subplot(gs[2, 2:4])

        # Title
        self.fig.suptitle("MPC-Based Smart Grid Control System",
                          fontsize=15, fontweight="bold", color=C["text"], y=0.97)

        # Keyboard handler
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _style(self, ax, off=False):
        ax.set_facecolor(C["panel"])
        if off:
            ax.axis("off")
        else:
            ax.tick_params(colors=C["dim"], labelsize=8)
            ax.grid(True, alpha=0.15, color=C["line"])
        for s in ax.spines.values():
            s.set_color(C["line"])

    # ── keyboard ─────────────────────────────────────────────────────────────
    def _on_key(self, event):
        if event.key == " ":
            self.paused = not self.paused
        elif event.key == "up":
            self.speed = min(600, int(self.speed * 1.5))
        elif event.key == "down":
            self.speed = max(5, int(self.speed / 1.5))
        elif event.key == "q":
            plt.close()

    # ── frame update ─────────────────────────────────────────────────────────
    def _update(self, _frame):
        if self.paused:
            return

        dt = self.speed  # minutes per tick
        self.sim_time += timedelta(minutes=dt)
        t = self.sim_time

        # Environment
        irr = self.env.irradiation(t)
        amb = self.env.ambient_temp(t)
        mod = self.env.module_temp(amb, irr)
        volt = self.env.voltage()

        # AI predictions
        solar = self.ai.predict_solar(t, irr, amb, mod)
        load = self.ai.predict_load(t, volt)

        # MPC
        source, _reason = self.mpc.decide(solar, load)
        self.mpc.step(solar, load, source, dt)
        soc = self.mpc.soc

        # Stats
        dt_h = dt / 60
        self.solar_kwh += solar * dt_h / 1000
        if source == "Grid":
            self.grid_kwh += load * dt_h / 1000

        # History
        self.t_hist.append(t)
        self.solar_hist.append(solar)
        self.load_hist.append(load)
        self.soc_hist.append(soc)
        self.src_hist.append(source)

        # Draw
        self._draw_flow(solar, load, source, soc)
        self._draw_timeseries()
        self._draw_pie()
        self._draw_battery(soc)
        self._draw_alert(source, solar, load, soc)

    # ── DRAWING ──────────────────────────────────────────────────────────────
    def _draw_flow(self, solar, load, source, soc):
        ax = self.ax_flow
        ax.clear()
        self._style(ax, off=True)

        time_str = self.sim_time.strftime("%Y-%m-%d  %H:%M")
        speed_str = f"Speed ×{self.speed}"
        status_str = "PAUSED" if self.paused else "RUNNING"

        ax.text(0.50, 0.95, f"Power Flow  —  {time_str}     [{speed_str}]  [{status_str}]",
                ha="center", va="top", fontsize=11, fontweight="bold", color=C["text"],
                transform=ax.transAxes)

        # Component boxes
        boxes = [
            (0.08, "SOLAR",   f"{solar:,.0f} W",    C["solar"],   source == "Solar"),
            (0.32, "BATTERY", f"{soc:.0f} %",        C["battery"], source == "Battery"),
            (0.56, "GRID",    "ACTIVE" if source == "Grid" else "STANDBY",
                                                      C["grid"],    source == "Grid"),
            (0.80, "LOAD",    f"{load:,.0f} W",      C["load"],    True),
        ]
        for cx, label, val, col, active in boxes:
            alpha = 0.35 if active else 0.12
            ec = col if active else C["dim"]
            rect = mpatches.FancyBboxPatch(
                (cx - 0.07, 0.22), 0.14, 0.52,
                boxstyle="round,pad=0.02", transform=ax.transAxes,
                facecolor=col, alpha=alpha, edgecolor=ec, linewidth=2.5)
            ax.add_patch(rect)
            ax.text(cx, 0.60, val, ha="center", va="center",
                    fontsize=12, fontweight="bold", color=C["text"],
                    transform=ax.transAxes)
            ax.text(cx, 0.30, label, ha="center", va="center",
                    fontsize=9, color=C["dim"], transform=ax.transAxes)

        # Arrow from active source → load
        src_x = {"Solar": 0.15, "Battery": 0.39, "Grid": 0.63}
        arrow_col = {"Solar": C["solar"], "Battery": C["battery"], "Grid": C["grid"]}
        ax.annotate("", xy=(0.73, 0.48), xytext=(src_x[source], 0.48),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", lw=3, color=arrow_col[source]))

        # Active-source banner
        ax.text(0.50, 0.08, f"Active: {source.upper()}",
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=arrow_col[source], transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=arrow_col[source],
                          alpha=0.15, edgecolor=arrow_col[source], linewidth=1.5))

    def _draw_timeseries(self):
        ax = self.ax_ts
        ax.clear()
        self._style(ax)
        if len(self.t_hist) < 2:
            return
        t = list(self.t_hist)
        ax.plot(t, list(self.solar_hist), color=C["solar"], lw=2, label="Solar")
        ax.plot(t, list(self.load_hist),  color=C["load"],  lw=2, label="Load")
        ax.fill_between(t, list(self.solar_hist), alpha=0.15, color=C["solar"])
        ax.fill_between(t, list(self.load_hist),  alpha=0.10, color=C["load"])
        ax.set_ylabel("Power (W)", fontsize=9, color=C["text"])
        ax.set_title("Generation vs Consumption", fontsize=10, fontweight="bold",
                     color=C["text"], pad=8)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.6,
                  facecolor=C["panel"], edgecolor=C["line"])
        ax.tick_params(axis="x", rotation=15)

    def _draw_pie(self):
        ax = self.ax_pie
        ax.clear()
        self._style(ax, off=True)
        if not self.src_hist:
            return
        srcs = list(self.src_hist)
        counts = [srcs.count(s) for s in ("Solar", "Battery", "Grid")]
        total = len(srcs)
        pcts = [c / total * 100 for c in counts]
        labels = [f"Solar\n{pcts[0]:.0f}%", f"Battery\n{pcts[1]:.0f}%", f"Grid\n{pcts[2]:.0f}%"]
        cols = [C["solar"], C["battery"], C["grid"]]

        # Filter out zero slices
        non_zero = [(l, p, c) for l, p, c in zip(labels, pcts, cols) if p > 0]
        if not non_zero:
            return
        labels_f, pcts_f, cols_f = zip(*non_zero)

        wedges, texts = ax.pie(pcts_f, labels=labels_f, colors=cols_f,
                               startangle=90, wedgeprops=dict(width=0.55, edgecolor=C["panel"]),
                               textprops=dict(color=C["text"], fontsize=8, fontweight="bold"))
        ax.set_title("Source Mix", fontsize=10, fontweight="bold",
                     color=C["text"], pad=8)

    def _draw_battery(self, soc):
        ax = self.ax_bat
        ax.clear()
        self._style(ax)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])

        # Background bar
        ax.barh(0, 100, height=0.7, color=C["dim"], alpha=0.15, edgecolor=C["line"])
        # Filled bar
        bar_col = C["danger"] if soc < 20 else C["warn"] if soc < 40 else C["battery"]
        ax.barh(0, soc, height=0.7, color=bar_col, alpha=0.8, edgecolor=bar_col, linewidth=1.5)

        # Threshold markers
        ax.axvline(20, color=C["danger"], ls="--", lw=1.5, alpha=0.6)
        ax.axvline(90, color=C["ok"],     ls="--", lw=1.5, alpha=0.6)

        # Label
        status = "CRITICAL" if soc < 20 else "LOW" if soc < 40 else "GOOD" if soc < 90 else "FULL"
        ax.text(50, 0, f"{soc:.1f}%  ({status})", ha="center", va="center",
                fontsize=14, fontweight="bold", color=C["text"])

        ax.set_xlabel("State of Charge (%)", fontsize=9, color=C["dim"])
        ax.set_title("Battery", fontsize=10, fontweight="bold", color=C["text"], pad=8)

        # Tiny stats
        ax.text(1, -0.55, f"Solar: {self.solar_kwh:.1f} kWh   Grid: {self.grid_kwh:.1f} kWh",
                fontsize=7, color=C["dim"], va="bottom")

    def _draw_alert(self, source, solar, load, soc):
        ax = self.ax_alert
        ax.clear()
        self._style(ax, off=True)

        atype, title, detail = generate_alert(source, solar, load, soc)
        col = {"ok": C["ok"], "warn": C["warn"], "danger": C["danger"]}[atype]

        # Card background
        rect = mpatches.FancyBboxPatch(
            (0.04, 0.10), 0.92, 0.78,
            boxstyle="round,pad=0.03", transform=ax.transAxes,
            facecolor=col, alpha=0.12, edgecolor=col, linewidth=2)
        ax.add_patch(rect)

        ax.text(0.50, 0.80, "SYSTEM STATUS", ha="center", va="center",
                fontsize=10, fontweight="bold", color=C["text"],
                transform=ax.transAxes)
        ax.text(0.50, 0.55, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=col,
                transform=ax.transAxes)
        ax.text(0.50, 0.32, detail, ha="center", va="center",
                fontsize=9, color=C["dim"], transform=ax.transAxes)

        # Battery sub-alert
        if soc < 20:
            ax.text(0.50, 0.15, "⚠ Battery critically low — grid charging",
                    ha="center", va="center", fontsize=8, color=C["danger"],
                    transform=ax.transAxes)
        elif soc > 95:
            ax.text(0.50, 0.15, "✓ Battery fully charged",
                    ha="center", va="center", fontsize=8, color=C["ok"],
                    transform=ax.transAxes)

    # ── run ──────────────────────────────────────────────────────────────────
    def run(self):
        self.init()
        self.ani = FuncAnimation(self.fig, self._update, interval=1000,
                                 cache_frame_data=False)
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="MPC Smart Grid Simulation Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Controls:  SPACE=pause  ↑↓=speed  Q=quit")
    parser.add_argument("--speed", type=int, default=60,
                        help="Simulated minutes per real second (default: 60)")
    args = parser.parse_args()
    Dashboard(speed=args.speed).run()


if __name__ == "__main__":
    main()
