import os
import sys
import json
from datetime import datetime
import numpy as np
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# ---- SETTINGS ----
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
WINDOW = 8

CHANNELS = ["O2 (%)", "Pressure (kPa)", "Temperature (°C)", "CO2 (ppm)", "Radiation (µSv/h)"]
ANOMALY_TYPES = ["Hull breach", "Scrubber failure", "Solar flare", "Heating failure", "Fire"]

# O2, pressure, temp, CO2, radiation, day cycle, activity, noise, severity min, severity max
PRESETS = {
    "Earth-like habitat": [20.9, 101.3, 21.0, 600, 0.2, 60, 0.5, 1.0, 1.0, 3.0],
    "Mars base":          [26.5, 70.0, 20.0, 1500, 12.0, 60, 0.6, 1.0, 1.0, 3.0],
    "Lunar outpost":      [30.0, 57.0, 19.0, 2000, 25.0, 60, 0.7, 1.2, 1.0, 3.0],
}
PARAM_LABELS = [
    "Baseline O2 (%)", "Baseline Pressure (kPa)", "Baseline Temperature (°C)", "Baseline CO2 (ppm)",
    "Baseline Radiation (µSv/h)", "Day Cycle Length (steps)", "Crew Activity", "Sensor Noise",
    "Severity Min", "Severity Max",
]
PARAM_RANGES = [(15, 35), (40, 110), (10, 30), (400, 5000), (0.1, 50), (20, 200), (0, 1), (0, 3), (0.1, 5), (0.1, 5)]
NUM_CONDITIONS = 8

# ---- MODEL FILES ----
def model_path(name):
    return os.path.join(MODEL_DIR, name)

def meta_path(name):
    return os.path.join(MODEL_DIR, name + ".json")

def list_saved_models():
    return sorted(f[:-4] for f in os.listdir(MODEL_DIR) if f.endswith(".zip"))

def read_meta(name):
    path = meta_path(name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def clean_name(name):
    name = name.strip().replace(" ", "_")
    return "".join(c for c in name if c.isalnum() or c in "_-")

# ---- GUI ----
def GUI():
    result = {}
    saved = list_saved_models()
    root = tk.Tk()
    root.title("Settlement Monitor")
    root.geometry("420x650")

    bottom = tk.Frame(root)
    bottom.pack(side="bottom", fill="x")
    canvas = tk.Canvas(root, highlightthickness=0)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    body = tk.Frame(canvas)
    body_window = canvas.create_window((0, 0), window=body, anchor="nw")
    body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda e: canvas.itemconfigure(body_window, width=e.width))

    def on_wheel(e):
        if e.num == 4:
            step = -1
        elif e.num == 5:
            step = 1
        elif sys.platform == "darwin":
            step = -e.delta
        else:
            step = -int(e.delta / 120)
        canvas.yview_scroll(step, "units")

    root.bind_all("<MouseWheel>", on_wheel)
    root.bind_all("<Button-4>", on_wheel)
    root.bind_all("<Button-5>", on_wheel)

    def section(text):
        tk.Label(body, text=text, font=("Arial", 9, "bold")).pack(anchor="w", padx=10, pady=(6, 0))

    def slider(label, f, t, var, resolution=0.01):
        tk.Label(body, text=label).pack(anchor="w", padx=10)
        tk.Scale(body, from_=f, to=t, resolution=resolution, variable=var, orient="horizontal").pack(fill="x", padx=10)

    section("Mode")
    mode_var = tk.StringVar(value="train")
    tk.Radiobutton(body, text="Train new model and save it", variable=mode_var, value="train").pack(anchor="w", padx=20)
    tk.Radiobutton(body, text="Load saved model", variable=mode_var, value="load").pack(anchor="w", padx=20)

    section("New model name")
    name_var = tk.StringVar(value=datetime.now().strftime("settlement_%Y%m%d_%H%M%S"))
    tk.Entry(body, textvariable=name_var).pack(fill="x", padx=10)

    section("Saved model")
    load_options = saved if saved else ["(none saved)"]
    load_var = tk.StringVar(value=load_options[-1])
    tk.OptionMenu(body, load_var, *load_options).pack(fill="x", padx=10)

    section("Timesteps per episode")
    ts_var = tk.IntVar(value=150)
    tk.Scale(body, from_=60, to=500, variable=ts_var, orient="horizontal").pack(fill="x", padx=10)

    section("PPO Training Timesteps (train mode only)")
    ppo_ts_var = tk.IntVar(value=300_000)
    tk.Scale(body, from_=50_000, to=1_000_000, resolution=10_000, variable=ppo_ts_var, orient="horizontal").pack(fill="x", padx=10)

    section("Test episodes to generate")
    episodes_var = tk.IntVar(value=10)
    tk.Scale(body, from_=1, to=50, variable=episodes_var, orient="horizontal").pack(fill="x", padx=10)
    guarantee_var = tk.BooleanVar(value=True)
    tk.Checkbutton(body, text="Every test episode contains an event", variable=guarantee_var).pack(anchor="w", padx=10)

    param_vars = [tk.DoubleVar() for _ in PARAM_LABELS]

    def apply_preset(val):
        for var, value in zip(param_vars, PRESETS[val]):
            var.set(value)

    section("Settlement type")
    style_var = tk.StringVar(value="Mars base")
    tk.OptionMenu(body, style_var, *PRESETS.keys(), command=apply_preset).pack(fill="x", padx=10)

    section("Settlement conditions")
    for lbl, (f, t), var in list(zip(PARAM_LABELS, PARAM_RANGES, param_vars))[:NUM_CONDITIONS]:
        slider(lbl, f, t, var)

    section("Event severity (1 = bad, 3 = catastrophic)")
    for lbl, (f, t), var in list(zip(PARAM_LABELS, PARAM_RANGES, param_vars))[NUM_CONDITIONS:]:
        slider(lbl, f, t, var)
    apply_preset("Mars base")

    def copy_settings():
        meta = read_meta(load_var.get())
        if meta is None or len(meta.get("params", [])) != len(PARAM_LABELS):
            messagebox.showinfo("No settings", "No compatible settings saved for that model.")
            return
        for var, value in zip(param_vars, meta["params"]):
            var.set(value)
        ts_var.set(meta["episode_length"])
        style_var.set(meta.get("style", style_var.get()))

    tk.Button(body, text="Copy settings from saved model", command=copy_settings).pack(pady=(8, 0))

    def on_submit():
        mode = mode_var.get()
        if mode == "load" and not saved:
            messagebox.showerror("No models", "There are no saved models yet. Train one first.")
            return
        name = clean_name(name_var.get()) if mode == "train" else load_var.get()
        if not name:
            messagebox.showerror("Bad name", "Give the model a name.")
            return
        if mode == "train" and name in saved:
            if not messagebox.askyesno("Overwrite?", f"'{name}' already exists. Overwrite it?"):
                return
        result["mode"] = mode
        result["model_name"] = name
        result["timesteps"] = ts_var.get()
        result["ppo_timesteps"] = ppo_ts_var.get()
        result["test_episodes"] = episodes_var.get()
        result["force_anomaly"] = guarantee_var.get()
        result["style"] = style_var.get()
        result["params"] = [v.get() for v in param_vars]
        root.destroy()

    tk.Button(bottom, text="Run", command=on_submit).pack(pady=10)
    root.mainloop()
    return result

# ---- SENSOR GENERATOR ----
def generate_settlement(num_steps, params, rng=None, force_anomaly=False):
    if rng is None:
        rng = np.random.default_rng()
    o2_b, p_b, t_b, co2_b, rad_b, cycle, activity, noise, sev_min, sev_max = params
    time = np.arange(num_steps)
    phase = 2 * np.pi * time / max(cycle, 1)
    day = np.sin(phase)
    work = np.sin(phase - 1.0)

    temp = t_b + 1.5 * day + rng.normal(0, 0.2 * noise, num_steps)
    co2 = co2_b * (1 + 0.3 * activity * work) + rng.normal(0, 20 * noise, num_steps)
    o2 = o2_b - 0.4 * activity * work + rng.normal(0, 0.05 * noise, num_steps)
    pressure = p_b * (1 + 0.003 * day) + rng.normal(0, 0.05 * noise, num_steps)
    radiation = rad_b * (1 + 0.1 * np.sin(phase / 3.7)) + rng.normal(0, 0.05 * rad_b * noise, num_steps)
    readings = np.column_stack([o2, pressure, temp, co2, radiation])

    labels = np.zeros(num_steps, dtype=np.float32)
    anomaly_starts = []
    anomaly_types = []

    # 35% of training episodes get an event; force_anomaly guarantees one
    if force_anomaly or rng.random() < 0.35:
        min_start = WINDOW + 5
        max_start = max(min_start, num_steps - 21)
        start = int(rng.integers(min_start, max_start + 1))
        length = min(int(rng.integers(8, 21)), num_steps - start)
        p = np.linspace(0, 1, length)
        s = slice(start, start + length)
        kind = ANOMALY_TYPES[int(rng.integers(len(ANOMALY_TYPES)))]
        sev = rng.uniform(min(sev_min, sev_max), max(sev_min, sev_max))
        jitter = lambda scale: rng.normal(0, scale, length)

        if kind == "Hull breach":
            drop = 1 - np.exp(-4 * sev * p)
            readings[s, 1] *= 1 - 0.9 * drop
            readings[s, 1] += jitter(2 * sev)
            readings[s, 2] -= 25 * sev * drop + np.abs(jitter(2))
            readings[s, 0] += jitter(1.5 * sev)
        elif kind == "Scrubber failure":
            runaway = (np.exp(3 * p) - 1) / (np.e ** 3 - 1)
            readings[s, 3] += co2_b * 5 * sev * runaway
            readings[s, 0] -= 6 * sev * runaway
            readings[s, 2] += 3 * runaway
        elif kind == "Solar flare":
            bursts = np.zeros(length)
            for _ in range(int(rng.integers(2, 5))):
                bursts += np.exp(-((p - rng.uniform(0, 1)) ** 2) / 0.005)
            readings[s, 4] *= 1 + 40 * sev * bursts
            glitch = rng.random((length, 4)) < 0.25 * np.clip(bursts, 0, 1)[:, None]
            readings[s, :4] += glitch * rng.normal(0, 8, (length, 4)) * channel_scales(params)[:4]
        elif kind == "Heating failure":
            readings[s, 2] -= 30 * sev * p
            readings[s, 2] += 6 * sev * np.abs(np.sin(6 * np.pi * p)) * (1 - p)
            readings[s, 1] *= 1 - 0.1 * sev * p
            readings[s, 0] += jitter(0.3)
        elif kind == "Fire":
            flame = p ** 2
            readings[s, 2] += 80 * sev * flame + np.abs(jitter(5 * sev)) * p
            readings[s, 3] += co2_b * 8 * sev * flame
            readings[s, 0] -= 10 * sev * flame
            readings[s, 1] *= np.where(p > 0.7, 1 - 0.5 * (p - 0.7) / 0.3, 1 + 0.15 * sev * p)

        labels[s] = 1
        anomaly_starts.append(start)
        anomaly_types.append(kind)

    readings[:, [0, 1, 3, 4]] = np.maximum(readings[:, [0, 1, 3, 4]], 0)
    return readings, labels, anomaly_starts, anomaly_types

# ---- NORMALISATION ----
def channel_scales(params):
    o2_b, p_b, t_b, co2_b, rad_b, cycle, activity = params[:7]
    return np.array([0.5, max(0.01 * p_b, 0.5), 2.0, max(0.3 * co2_b * max(activity, 0.2), 50.0), max(0.2 * rad_b, 0.05)])

def normalise(readings, params):
    data = (readings - np.array(params[:5])) / channel_scales(params)
    data = np.nan_to_num(data, nan=0.0, posinf=10.0, neginf=-10.0)
    return np.clip(data, -10.0, 10.0).astype(np.float32)

# ---- ENVIRONMENT ----
class SettlementEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, params, num_steps=150, window=WINDOW):
        super().__init__()
        self.params = params
        self.num_steps = num_steps
        self.window = window
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(window, len(CHANNELS)), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        readings, labels, anomaly_starts, anomaly_types = generate_settlement(self.num_steps, self.params, self.np_random)
        self.labels = labels
        self.anomaly_starts = anomaly_starts
        self.data = normalise(readings, self.params)
        self.i = self.window - 1
        return self.get_observation(), {}

    def get_observation(self):
        return self.data[self.i - self.window + 1:self.i + 1].copy()

    def step(self, action):
        correct = int(self.labels[self.i])
        if correct == 1 and action == 1:
            reward = 10.0
        elif correct == 1 and action == 0:
            reward = -10.0
        elif correct == 0 and action == 1:
            reward = -5.0
        else:
            reward = 1.0
        self.i += 1
        terminated = self.i >= self.num_steps
        if terminated:
            observation = np.zeros((self.window, len(CHANNELS)), dtype=np.float32)
        else:
            observation = self.get_observation()
        return observation, reward, terminated, False, {}

# ---- TRAIN AND SAVE ----
def train_and_save(settings):
    name = settings["model_name"]
    print(f"\n========================================\n Training PPO model '{name}'\n========================================\n")
    env = SettlementEnv(params=settings["params"], num_steps=settings["timesteps"], window=WINDOW)
    model = PPO(
        "MlpPolicy", env,
        learning_rate=0.0001, n_steps=256, batch_size=64, ent_coef=0.001,
        gamma=0.99, clip_range=0.2, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[64, 64]), verbose=1, device="cpu",
    )
    model.learn(total_timesteps=settings["ppo_timesteps"])
    model.save(model_path(name))
    meta = {
        "name": name,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "style": settings["style"],
        "params": settings["params"],
        "channels": CHANNELS,
        "episode_length": settings["timesteps"],
        "ppo_timesteps": settings["ppo_timesteps"],
        "window": WINDOW,
    }
    with open(meta_path(name), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nSaved model to   {model_path(name)}.zip")
    print(f"Saved settings to {meta_path(name)}\n")
    return model

# ---- LOAD ----
def load_saved(name):
    model = PPO.load(model_path(name), device="cpu")
    expected = (WINDOW, len(CHANNELS))
    if model.observation_space.shape != expected:
        raise SystemExit(
            f"'{name}' was trained on different data (input shape {model.observation_space.shape}, "
            f"this script needs {expected}). Train a new model."
        )
    meta = read_meta(name)
    print(f"\n========================================\n Loaded model '{name}'\n========================================")
    if meta:
        print("Trained at    :", meta["trained_at"])
        print("Trained type  :", meta["style"])
        print("Trained params:", [round(p, 2) for p in meta["params"]])
        print("PPO timesteps :", meta["ppo_timesteps"])
    else:
        print("(No settings file found for this model)")
    print()
    return model, meta

# ---- RUN ONE EPISODE ----
def run_episode(model, params, num_steps, rng, force_anomaly):
    readings, labels, starts, types = generate_settlement(num_steps, params, rng, force_anomaly=force_anomaly)
    data = normalise(readings, params)
    windows = np.stack([data[i - WINDOW + 1:i + 1] for i in range(WINDOW - 1, num_steps)])
    actions, _ = model.predict(windows, deterministic=True)
    flags = np.zeros(num_steps, dtype=bool)
    flags[WINDOW - 1:] = np.asarray(actions).reshape(-1) == 1
    return {"readings": readings, "truth": labels == 1, "flags": flags, "starts": starts, "types": types}

# ---- METRICS ----
def compute_metrics(flags, truth):
    tp = int(np.sum(flags & truth))
    fp = int(np.sum(flags & ~truth))
    fn = int(np.sum(~flags & truth))
    tn = int(np.sum(~flags & ~truth))
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "accuracy": (tp + tn) / max(len(flags), 1),
    }

def detection_delay(ep):
    if not ep["starts"]:
        return None
    event_steps = np.where(ep["truth"])[0]
    caught = event_steps[ep["flags"][event_steps]]
    if len(caught) == 0:
        return None
    return int(caught[0] - ep["starts"][0])

# ---- EVALUATE ----
def evaluate(model, settings):
    rng = np.random.default_rng()
    episodes = []
    by_type = {kind: [0, 0] for kind in ANOMALY_TYPES}
    print("==============================\n PPO ANOMALY DETECTION RESULTS\n==============================")

    for n in range(settings["test_episodes"]):
        ep = run_episode(model, settings["params"], settings["timesteps"], rng, settings["force_anomaly"])
        episodes.append(ep)
        m = compute_metrics(ep["flags"], ep["truth"])
        delay = detection_delay(ep)
        if ep["starts"]:
            kind = ep["types"][0]
            by_type[kind][1] += 1
            if delay is None:
                event_text = f"{kind} at {ep['starts'][0]}, MISSED"
            else:
                by_type[kind][0] += 1
                event_text = f"{kind} at {ep['starts'][0]}, caught after {delay} steps"
        else:
            event_text = "no event"
        print(f"Episode {n + 1:>2}: {event_text}, {m['fp']} false alarm steps")

    all_flags = np.concatenate([ep["flags"] for ep in episodes])
    all_truth = np.concatenate([ep["truth"] for ep in episodes])
    m = compute_metrics(all_flags, all_truth)
    with_event = [ep for ep in episodes if ep["starts"]]
    caught = [d for d in (detection_delay(ep) for ep in with_event) if d is not None]

    print()
    print("True positives :", m["tp"])
    print("False positives:", m["fp"])
    print("Missed steps   :", m["fn"])
    print("True negatives :", m["tn"])
    print()
    print("Accuracy       :", round(m["accuracy"], 3))
    print("Precision      :", round(m["precision"], 3))
    print("Recall         :", round(m["recall"], 3))
    print()
    if with_event:
        print(f"Events caught  : {len(caught)} / {len(with_event)}")
        if caught:
            print(f"Avg delay      : {np.mean(caught):.1f} steps")
        print("\nBy event type:")
        for kind, (c, total) in by_type.items():
            if total:
                print(f"  {kind:<17}: {c} / {total} caught")
    print()
    return episodes

# ---- PLOT ----
def plot_episode(ep, title):
    t = np.arange(len(ep["readings"]))
    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(11, 9), sharex=True)
    for c, (ax, name) in enumerate(zip(axes, CHANNELS)):
        ax.plot(t, ep["readings"][:, c], lw=1)
        ax.set_ylabel(name, fontsize=8)
        if ep["truth"].any():
            event_t = t[ep["truth"]]
            ax.axvspan(event_t[0], event_t[-1], color="red", alpha=0.15, label="Actual event" if c == 0 else None)
        ax.scatter(t[ep["flags"]], ep["readings"][ep["flags"], c], color="black", s=20, marker="x",
                   label="PPO flagged" if c == 0 else None, zorder=5)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("Time Step (t)")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# ---- MAIN ----
settings = GUI()
if not settings:
    raise SystemExit("Window closed without running.")

if settings["mode"] == "train":
    model = train_and_save(settings)
else:
    model, meta = load_saved(settings["model_name"])

episodes = evaluate(model, settings)
last = episodes[-1]
event_name = last["types"][0] if last["types"] else "no event"
plot_episode(last, f"Settlement Anomaly Detection: {settings['model_name']} (last episode: {event_name})")