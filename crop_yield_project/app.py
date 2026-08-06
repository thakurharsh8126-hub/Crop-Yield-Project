
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from model import CropYieldModel, DATA_PATH


if not os.path.exists(DATA_PATH):
    import generate_data  # noqa: F401  (running this module generates the CSV)


BG = "#0f172a"
PANEL = "#1e293b"
ACCENT = "#22c55e"
ACCENT2 = "#38bdf8"
TEXT = "#e2e8f0"
SUBTEXT = "#94a3b8"
FONT = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 11, "bold")
FONT_TITLE = ("Segoe UI", 16, "bold")

plt.rcParams.update({
    "figure.facecolor": PANEL,
    "axes.facecolor": PANEL,
    "axes.edgecolor": SUBTEXT,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": SUBTEXT,
    "ytick.color": SUBTEXT,
    "axes.titlecolor": TEXT,
    "grid.color": "#334155",
})

PIE_COLORS = ["#22c55e", "#38bdf8", "#f59e0b", "#ef4444", "#a78bfa", "#f472b6"]


class CropYieldApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Crop Yield Prediction System")
        self.root.geometry("1300x800")
        self.root.configure(bg=BG)
        self.root.minsize(1100, 700)

        self.cy_model = CropYieldModel()
        self.metrics = self.cy_model.train()
        self.df = self.cy_model.df

        self._build_layout()

    
    def _build_layout(self):
        # Header
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=20, pady=(15, 5))

        tk.Label(
            header, text="🌾 Crop Yield Prediction System", font=FONT_TITLE,
            bg=BG, fg=TEXT
        ).pack(side="left")

        tk.Label(
            header,
            text=f"Model R²: {self.metrics['r2']:.3f}   |   MAE: {self.metrics['mae']:.2f} t/ha   |   RMSE: {self.metrics['rmse']:.2f} t/ha",
            font=FONT, bg=BG, fg=ACCENT2
        ).pack(side="right")

        # Notebook (tabs)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure(
            "TNotebook.Tab", background=PANEL, foreground=TEXT,
            padding=[16, 8], font=FONT_BOLD
        )
        style.map("TNotebook.Tab", background=[("selected", ACCENT)],
                  foreground=[("selected", "#0f172a")])

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=10)

        self.predict_tab = tk.Frame(self.notebook, bg=BG)
        self.dashboard_tab = tk.Frame(self.notebook, bg=BG)
        self.model_tab = tk.Frame(self.notebook, bg=BG)

        self.notebook.add(self.predict_tab, text="  🔮 Predict Yield  ")
        self.notebook.add(self.dashboard_tab, text="  📊 Data Dashboard  ")
        self.notebook.add(self.model_tab, text="  🧠 Model Insights  ")

        self._build_predict_tab()
        self._build_dashboard_tab()
        self._build_model_tab()

    
    def _build_predict_tab(self):
        container = tk.Frame(self.predict_tab, bg=BG)
        container.pack(fill="both", expand=True)

        # Left: form (fixed width panel that holds a scrollable area on top
        # and a pinned action bar -- button + result -- at the bottom, so the
        # button is ALWAYS visible no matter how tall the form content is
        # or how small the window/screen is).
        form_outer = tk.Frame(container, bg=PANEL)
        form_outer.pack(side="left", fill="y", padx=(0, 15), pady=10)
        form_outer.pack_propagate(False)
        form_outer.configure(width=340)

        # --- Pinned action bar (packed BOTTOM first so it always reserves
        # its space, regardless of how much content is above it) ---
        action_bar = tk.Frame(form_outer, bg=PANEL, padx=25, pady=15)
        action_bar.pack(side="bottom", fill="x")

        predict_btn = tk.Button(
            action_bar, text="Predict Yield ➜", font=FONT_BOLD, bg=ACCENT,
            fg="#052e16", activebackground="#16a34a", relief="flat",
            cursor="hand2", command=self.run_prediction, pady=10
        )
        predict_btn.pack(fill="x")

        # --- Large result box (shown after prediction) ---
        self.result_box = tk.Frame(action_bar, bg="#0f2e1a", bd=0,
                                    highlightthickness=2, highlightbackground=ACCENT)
        self.result_box.pack(fill="x", pady=(12, 0))

        inner = tk.Frame(self.result_box, bg="#0f2e1a", padx=16, pady=14)
        inner.pack(fill="x")

        self.result_caption_lbl = tk.Label(
            inner, text="PREDICTED YIELD", font=("Segoe UI", 9, "bold"),
            bg="#0f2e1a", fg=SUBTEXT
        )
        self.result_caption_lbl.pack(anchor="w")

        self.result_value_lbl = tk.Label(
            inner, text="—", font=("Segoe UI", 34, "bold"),
            bg="#0f2e1a", fg=ACCENT
        )
        self.result_value_lbl.pack(anchor="w", pady=(2, 0))

        self.result_unit_lbl = tk.Label(
            inner, text="tons / hectare", font=("Segoe UI", 10),
            bg="#0f2e1a", fg=SUBTEXT
        )
        self.result_unit_lbl.pack(anchor="w", pady=(0, 8))

        self.result_badge_lbl = tk.Label(
            inner, text="", font=("Segoe UI", 10, "bold"),
            bg="#0f2e1a", fg=ACCENT, padx=8, pady=3
        )
        self.result_badge_lbl.pack(anchor="w")

        self.result_interp_lbl = tk.Label(
            inner, text="Enter field conditions and click Predict Yield to see\n"
                        "your result and an interpretation here.",
            font=("Segoe UI", 9), bg="#0f2e1a", fg=TEXT,
            wraplength=280, justify="left"
        )
        self.result_interp_lbl.pack(anchor="w", pady=(8, 0))

        # --- Scrollable form area (everything else) ---
        scroll_container = tk.Frame(form_outer, bg=PANEL)
        scroll_container.pack(side="top", fill="both", expand=True)

        canvas = tk.Canvas(scroll_container, bg=PANEL, highlightthickness=0)
        scrollbar = tk.Scrollbar(scroll_container, orient="vertical", command=canvas.yview)
        form_panel = tk.Frame(canvas, bg=PANEL, padx=25, pady=20)

        form_panel.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas_window = canvas.create_window((0, 0), window=form_panel, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _resize_inner(event):
            canvas.itemconfig(canvas_window, width=event.width)

        canvas.bind("<Configure>", _resize_inner)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        tk.Label(form_panel, text="Enter Field Conditions", font=FONT_BOLD,
                 bg=PANEL, fg=TEXT).pack(anchor="w", pady=(0, 15))

        self.vars = {}

        def add_slider(label, key, frm, to, default, unit=""):
            tk.Label(form_panel, text=f"{label} ({unit})" if unit else label,
                     font=FONT, bg=PANEL, fg=SUBTEXT).pack(anchor="w", pady=(10, 0))
            var = tk.DoubleVar(value=default)
            val_lbl = tk.Label(form_panel, text=f"{default:.1f}", font=FONT_BOLD,
                                bg=PANEL, fg=ACCENT2)
            val_lbl.pack(anchor="e")

            def on_change(v):
                val_lbl.config(text=f"{float(v):.1f}")

            scale = tk.Scale(
                form_panel, variable=var, from_=frm, to=to, orient="horizontal",
                length=280, bg=PANEL, fg=TEXT, troughcolor="#334155",
                highlightthickness=0, showvalue=False, command=on_change,
                activebackground=ACCENT
            )
            scale.pack(fill="x")
            self.vars[key] = var

        add_slider("Rainfall", "rainfall", 200, 3000, 1200, "mm")
        add_slider("Temperature", "temperature", 10, 45, 25, "°C")
        add_slider("Humidity", "humidity", 20, 100, 60, "%")
        add_slider("Fertilizer Usage", "fertilizer", 0, 300, 100, "kg/ha")

        tk.Label(form_panel, text="Soil Type", font=FONT, bg=PANEL,
                 fg=SUBTEXT).pack(anchor="w", pady=(15, 3))
        self.soil_var = tk.StringVar(value=self.cy_model.get_soil_types()[0])
        soil_menu = ttk.Combobox(form_panel, textvariable=self.soil_var,
                                  values=self.cy_model.get_soil_types(),
                                  state="readonly", font=FONT)
        soil_menu.pack(fill="x")

        tk.Label(form_panel, text="Crop Type", font=FONT, bg=PANEL,
                 fg=SUBTEXT).pack(anchor="w", pady=(15, 3))
        self.crop_var = tk.StringVar(value=self.cy_model.get_crop_types()[0])
        crop_menu = ttk.Combobox(form_panel, textvariable=self.crop_var,
                                  values=self.cy_model.get_crop_types(),
                                  state="readonly", font=FONT)
        crop_menu.pack(fill="x", pady=(0, 10))

        # Right: chart panel
        chart_panel = tk.Frame(container, bg=PANEL)
        chart_panel.pack(side="left", fill="both", expand=True, pady=10)

        self.pred_fig = Figure(figsize=(7, 6), dpi=100)
        self.pred_canvas = FigureCanvasTkAgg(self.pred_fig, master=chart_panel)
        self.pred_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        self._draw_prediction_placeholder()

    def _draw_prediction_placeholder(self):
        self.pred_fig.clear()
        ax1 = self.pred_fig.add_subplot(211)
        ax2 = self.pred_fig.add_subplot(212)

        # Bar: contribution factors (feature importance) as placeholder
        fi = self.cy_model.feature_importance
        ax1.barh(fi.index[::-1], fi.values[::-1], color=PIE_COLORS)
        ax1.set_title("Feature Importance (Model-wide)")
        ax1.set_xlabel("Importance")

        # Pie: crop distribution in dataset
        counts = self.df["Crop_Type"].value_counts()
        ax2.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=PIE_COLORS, textprops={"color": TEXT})
        ax2.set_title("Crop Type Distribution in Training Data")

        self.pred_fig.tight_layout()
        self.pred_canvas.draw()

    def run_prediction(self):
        try:
            rainfall = self.vars["rainfall"].get()
            temperature = self.vars["temperature"].get()
            humidity = self.vars["humidity"].get()
            fertilizer = self.vars["fertilizer"].get()
            soil = self.soil_var.get()
            crop = self.crop_var.get()

            pred = self.cy_model.predict(
                rainfall, temperature, humidity, soil, fertilizer, crop
            )
            self._update_result_box(pred, crop)
            self._draw_prediction_result(pred, crop)
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    def _update_result_box(self, pred, crop):
        """Fill the large result box with the prediction and a plain-language
        interpretation, color-coded by how the yield compares to the rest
        of the dataset for that crop."""
        crop_yields = self.df[self.df["Crop_Type"] == crop]["Yield_tons_per_ha"]
        avg = crop_yields.mean()
        percentile = (crop_yields < pred).mean() * 100  # % of samples below this prediction

        if percentile >= 75:
            level = "Excellent"
            color = "#22c55e"       # green
            box_bg = "#0f2e1a"
            msg = (f"This is an excellent yield — better than {percentile:.0f}% of "
                   f"{crop} results in the dataset. Current rainfall, temperature, "
                   f"humidity, fertilizer and soil conditions are working well together.")
        elif percentile >= 50:
            level = "Good"
            color = "#38bdf8"       # blue
            box_bg = "#0c2a3d"
            msg = (f"This is a good yield — above the average of "
                   f"{avg:.2f} t/ha for {crop} and better than {percentile:.0f}% of "
                   f"comparable results. There's still some room to optimize.")
        elif percentile >= 25:
            level = "Below Average"
            color = "#f59e0b"       # amber
            box_bg = "#3a2a0c"
            msg = (f"This yield is below the {crop} average of {avg:.2f} t/ha "
                   f"(higher than only {percentile:.0f}% of results). Consider "
                   f"adjusting fertilizer usage or checking if soil type suits {crop}.")
        else:
            level = "Poor"
            color = "#ef4444"       # red
            box_bg = "#3a0f0f"
            msg = (f"This yield is significantly below the {crop} average of "
                   f"{avg:.2f} t/ha — only {percentile:.0f}% of results are this low "
                   f"or lower. Rainfall, temperature or fertilizer levels may be "
                   f"unfavorable for {crop} under these conditions.")

        # Update box + text colors
        self.result_box.configure(highlightbackground=color)
        for widget in (self.result_box,):
            widget.configure(bg=box_bg)
        self.result_caption_lbl.configure(bg=box_bg, text=f"PREDICTED YIELD — {crop.upper()}")
        self.result_value_lbl.configure(bg=box_bg, fg=color, text=f"{pred:.2f}")
        self.result_unit_lbl.configure(bg=box_bg, text="tons / hectare")
        self.result_badge_lbl.configure(
            bg=color, fg="#0f172a", text=f"  {level}  "
        )
        self.result_interp_lbl.configure(bg=box_bg, fg=TEXT, text=msg)

        # propagate bg to the inner container too
        for child in self.result_box.winfo_children():
            child.configure(bg=box_bg)
            for grandchild in child.winfo_children():
                if grandchild is not self.result_badge_lbl:
                    grandchild.configure(bg=box_bg)

    def _draw_prediction_result(self, pred, crop):
        self.pred_fig.clear()
        ax1 = self.pred_fig.add_subplot(211)
        ax2 = self.pred_fig.add_subplot(212)

        # Gauge-like bar comparing prediction to crop's average yield range
        crop_df = self.df[self.df["Crop_Type"] == crop]["Yield_tons_per_ha"]
        avg = crop_df.mean()
        mx = crop_df.max()

        bars = ax1.bar(["Your Prediction", f"{crop} Avg (dataset)", f"{crop} Max (dataset)"],
                        [pred, avg, mx], color=[ACCENT, ACCENT2, "#f59e0b"])
        ax1.set_title(f"Predicted Yield vs {crop} Benchmarks")
        ax1.set_ylabel("tons/hectare")
        for b in bars:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width() / 2, h, f"{h:.2f}",
                      ha="center", va="bottom", color=TEXT, fontsize=9)

        # Pie: relative contribution of factors for this specific prediction
        # (normalized feature importance x this sample's magnitude, illustrative)
        fi = self.cy_model.feature_importance
        ax2.pie(fi.values, labels=fi.index, autopct="%1.1f%%",
                colors=PIE_COLORS, textprops={"color": TEXT, "fontsize": 8})
        ax2.set_title("Relative Influence of Each Factor")

        self.pred_fig.tight_layout()
        self.pred_canvas.draw()

    
    def _build_dashboard_tab(self):
        frame = tk.Frame(self.dashboard_tab, bg=BG)
        frame.pack(fill="both", expand=True)

        fig = Figure(figsize=(12, 8), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        df = self.df

        # 1. Pie chart - Soil type distribution
        ax1 = fig.add_subplot(231)
        soil_counts = df["Soil_Type"].value_counts()
        ax1.pie(soil_counts.values, labels=soil_counts.index, autopct="%1.0f%%",
                colors=PIE_COLORS, textprops={"color": TEXT, "fontsize": 8})
        ax1.set_title("Soil Type Distribution", fontsize=10)

        # 2. Bar chart - Avg yield by crop
        ax2 = fig.add_subplot(232)
        avg_yield = df.groupby("Crop_Type")["Yield_tons_per_ha"].mean().sort_values()
        ax2.barh(avg_yield.index, avg_yield.values, color=ACCENT2)
        ax2.set_title("Avg Yield by Crop", fontsize=10)
        ax2.set_xlabel("tons/ha", fontsize=8)

        # 3. Scatter - Rainfall vs Yield
        ax3 = fig.add_subplot(233)
        ax3.scatter(df["Rainfall_mm"], df["Yield_tons_per_ha"], s=6, alpha=0.4, color=ACCENT)
        ax3.set_title("Rainfall vs Yield", fontsize=10)
        ax3.set_xlabel("Rainfall (mm)", fontsize=8)
        ax3.set_ylabel("Yield (t/ha)", fontsize=8)

        # 4. Scatter - Temperature vs Yield
        ax4 = fig.add_subplot(234)
        ax4.scatter(df["Temperature_C"], df["Yield_tons_per_ha"], s=6, alpha=0.4, color="#f59e0b")
        ax4.set_title("Temperature vs Yield", fontsize=10)
        ax4.set_xlabel("Temperature (°C)", fontsize=8)
        ax4.set_ylabel("Yield (t/ha)", fontsize=8)

        # 5. Bar - Avg yield by soil type
        ax5 = fig.add_subplot(235)
        soil_yield = df.groupby("Soil_Type")["Yield_tons_per_ha"].mean().sort_values()
        ax5.bar(soil_yield.index, soil_yield.values, color="#a78bfa")
        ax5.set_title("Avg Yield by Soil Type", fontsize=10)
        ax5.tick_params(axis="x", rotation=30, labelsize=7)

        # 6. Histogram - Yield distribution
        ax6 = fig.add_subplot(236)
        ax6.hist(df["Yield_tons_per_ha"], bins=25, color=ACCENT, edgecolor=BG)
        ax6.set_title("Yield Distribution", fontsize=10)
        ax6.set_xlabel("tons/ha", fontsize=8)

        fig.tight_layout()
        canvas.draw()

    
    def _build_model_tab(self):
        frame = tk.Frame(self.model_tab, bg=BG)
        frame.pack(fill="both", expand=True)

        info = tk.Label(
            frame,
            text=f"Random Forest Regressor  |  Trees: 200  |  Max Depth: 12  |  "
                 f"Test R²: {self.metrics['r2']:.3f}  |  MAE: {self.metrics['mae']:.2f}  |  RMSE: {self.metrics['rmse']:.2f}",
            font=FONT_BOLD, bg=BG, fg=TEXT
        )
        info.pack(pady=(10, 0))

        fig = Figure(figsize=(12, 6), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        # Feature importance bar
        ax1 = fig.add_subplot(121)
        fi = self.cy_model.feature_importance.sort_values()
        ax1.barh(fi.index, fi.values, color=PIE_COLORS)
        ax1.set_title("Feature Importance")
        ax1.set_xlabel("Importance Score")

        # Predicted vs Actual scatter
        ax2 = fig.add_subplot(122)
        ax2.scatter(self.cy_model.y_test, self.cy_model.preds, s=10, alpha=0.5, color=ACCENT2)
        lims = [0, max(self.cy_model.y_test.max(), self.cy_model.preds.max()) + 0.5]
        ax2.plot(lims, lims, color="#ef4444", linestyle="--", linewidth=1.5, label="Perfect Prediction")
        ax2.set_xlim(lims)
        ax2.set_ylim(lims)
        ax2.set_xlabel("Actual Yield (t/ha)")
        ax2.set_ylabel("Predicted Yield (t/ha)")
        ax2.set_title("Predicted vs Actual (Test Set)")
        ax2.legend(fontsize=8)

        fig.tight_layout()
        canvas.draw()


def main():
    root = tk.Tk()
    app = CropYieldApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
