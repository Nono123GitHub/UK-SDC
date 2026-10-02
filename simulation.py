import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

DAMAGE_TYPES = ["Dent", "Puncture", "Sealant failure", "Circuit failure", "Hull rupture"]
SENSORS = ["Radiation", "Pressure", "Oxygen", "CO2", "Temperature", "Humidity"]
UNITS = ["mSv/h", "kPa", "%", "%", "C", "%"]

NORMAL_CENTER = np.array([0.03, 100.0, 21.5, 0.25, 22.0, 45.0])
NORMAL_HALF_WIDTH = np.array([0.02, 5.0, 2.0, 0.25, 4.0, 15.0])
NORMAL_LOW = NORMAL_CENTER - NORMAL_HALF_WIDTH
NORMAL_HIGH = NORMAL_CENTER + NORMAL_HALF_WIDTH
SENSOR_MAX = np.array([10.0, 300.0, 100.0, 20.0, 80.0, 100.0])

SIGNATURES = np.array([
    [0.8, -0.3, 0.0, 0.0, -0.3, 0.0],
    [0.5, -2.0, -1.0, 0.5, -1.0, -1.0],
    [0.0, -1.0, -0.5, 0.3, -0.5, -1.0],
    [0.0, 0.2, -1.0, 1.5, 2.0, -0.5],
    [2.0, -3.0, -2.0, -0.5, -2.5, -2.0],
])

DEVIATION_SCALE = 1.6
N_LEVELS = 5
N_SENSORS = len(SENSORS)
N_DAMAGE = len(DAMAGE_TYPES)

# Default run lengths (the original script used 6,000 training and 5,000 evaluation steps)
DEFAULT_TRAIN_STEPS = 200_000
DEFAULT_EVAL_STEPS = 50_000
# The learning rate shrinks as 1 / (1 + updates / DECAY). A fixed rate keeps bouncing between the
# overlapping damage types forever, so longer runs only help if the step size is allowed to shrink.
# Set to 0 to keep the original fixed learning rate.
DEFAULT_LR_DECAY_STEPS = 20_000


def symptom_levels(readings):
    deviation = (readings - NORMAL_CENTER) / NORMAL_HALF_WIDTH
    magnitude = np.abs(deviation)
    level = np.where(magnitude <= 1.0, 0, np.where(magnitude <= 2.0, 1, 2))
    return (level * np.sign(deviation)).astype(int)


def symptom_features(readings):
    levels = symptom_levels(readings)
    features = np.zeros(N_SENSORS * N_LEVELS)
    for sensor_index, level in enumerate(levels):
        features[sensor_index * N_LEVELS + level + 2] = 1.0
    return features


class LunarBaseEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n_robots=4, episode_length=200, noise_std=0.35, fault_prob=0.05,
                 solar_event_prob=0.12, dust_event_prob=0.10):
        super().__init__()
        self.n_robots = n_robots
        self.episode_length = episode_length
        self.noise_std = noise_std
        self.fault_prob = fault_prob
        self.solar_event_prob = solar_event_prob
        self.dust_event_prob = dust_event_prob
        self.action_space = spaces.Discrete(N_DAMAGE)
        self.observation_space = spaces.Box(
            low=np.zeros(N_SENSORS, dtype=np.float32),
            high=SENSOR_MAX.astype(np.float32),
            dtype=np.float32,
        )
        self.normal_low = NORMAL_LOW
        self.normal_high = NORMAL_HIGH
        self.current_damage = 0
        self.step_count = 0

    def _sense(self):
        severity = self.np_random.uniform(0.5, 1.7)
        jitter = self.np_random.normal(0.0, 0.25, size=N_SENSORS)
        deviation = SIGNATURES[self.current_damage] * DEVIATION_SCALE * severity + jitter
        if self.np_random.random() < self.solar_event_prob:
            deviation[0] += self.np_random.uniform(2.0, 5.0)
        if self.np_random.random() < self.dust_event_prob:
            deviation[4] += self.np_random.uniform(-2.0, -1.0)
            deviation[5] += self.np_random.uniform(-1.5, 1.5)
        true_values = NORMAL_CENTER + deviation * NORMAL_HALF_WIDTH
        robot_readings = true_values + self.np_random.normal(
            0.0, self.noise_std, size=(self.n_robots, N_SENSORS)
        ) * NORMAL_HALF_WIDTH
        faults = self.np_random.random((self.n_robots, N_SENSORS)) < self.fault_prob
        garbage = self.np_random.uniform(0.0, SENSOR_MAX, size=(self.n_robots, N_SENSORS))
        robot_readings = np.where(faults, garbage, robot_readings)
        fused = np.median(robot_readings, axis=0)
        return np.clip(fused, 0.0, SENSOR_MAX).astype(np.float32)

    def _new_damage(self):
        self.current_damage = int(self.np_random.integers(0, N_DAMAGE))
        return self._sense()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        observation = self._new_damage()
        return observation, {"true_damage": self.current_damage}

    def step(self, action):
        true_damage = self.current_damage
        correct = int(action) == true_damage
        reward = 1.0 if correct else -1.0
        self.step_count += 1
        truncated = self.step_count >= self.episode_length
        observation = self._new_damage()
        info = {"true_damage": true_damage, "correct": correct}
        return observation, reward, False, truncated, info


class SymptomAgent:
    def __init__(self, learning_rate=0.15, lr_decay_steps=DEFAULT_LR_DECAY_STEPS):
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.lr_decay_steps = lr_decay_steps
        self.updates = 0
        self.weights = np.zeros((N_DAMAGE, N_SENSORS * N_LEVELS))
        for damage_index in range(N_DAMAGE):
            expected = NORMAL_CENTER + SIGNATURES[damage_index] * DEVIATION_SCALE * NORMAL_HALF_WIDTH
            self.weights[damage_index] = symptom_features(expected)

    def act(self, observation):
        features = symptom_features(observation)
        return int(np.argmax(self.weights @ features))

    def learn(self, observation, action, true_damage):
        if action == true_damage:
            return
        features = symptom_features(observation)
        self.updates += 1
        if self.lr_decay_steps > 0:
            self.learning_rate = self.initial_learning_rate / (1.0 + self.updates / self.lr_decay_steps)
        self.weights[true_damage] += self.learning_rate * features
        self.weights[action] -= self.learning_rate * features


def run_training(env, agent, steps, seed):
    observation, _ = env.reset(seed=seed)
    outcomes = np.zeros(steps)
    for t in range(steps):
        action = agent.act(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        agent.learn(observation, action, info["true_damage"])
        outcomes[t] = 1.0 if info["correct"] else 0.0
        observation = next_observation
        if terminated or truncated:
            observation, _ = env.reset()
    return outcomes


def run_evaluation(env, agent, steps, seed):
    confusion = np.zeros((N_DAMAGE, N_DAMAGE), dtype=int)
    symptom_sum = np.zeros((N_DAMAGE, N_SENSORS))
    symptom_count = np.zeros(N_DAMAGE)
    observation, _ = env.reset(seed=seed)
    for _ in range(steps):
        action = agent.act(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        true_damage = info["true_damage"]
        confusion[true_damage, action] += 1
        symptom_sum[true_damage] += symptom_levels(observation)
        symptom_count[true_damage] += 1
        observation = next_observation
        if terminated or truncated:
            observation, _ = env.reset()
    mean_symptoms = symptom_sum / np.maximum(symptom_count[:, None], 1)
    return confusion, mean_symptoms


def rolling_mean(values, window):
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def build_report(confusion, mean_symptoms, training_outcomes, out_file):
    row_totals = confusion.sum(axis=1, keepdims=True)
    percent = confusion / np.maximum(row_totals, 1) * 100.0
    per_class = np.diag(confusion) / np.maximum(row_totals.flatten(), 1) * 100.0
    overall = np.trace(confusion) / confusion.sum() * 100.0
    short_names = ["Dent", "Puncture", "Sealant", "Circuit", "Hull"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Lunar base autonomous repair network: damage diagnosis from sensor symptoms", fontsize=14)

    ax = axes[0, 0]
    image = ax.imshow(percent, cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(range(N_DAMAGE))
    ax.set_yticks(range(N_DAMAGE))
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_yticklabels(DAMAGE_TYPES)
    ax.set_xlabel("Diagnosed as")
    ax.set_ylabel("Actual damage")
    ax.set_title("Confusion matrix (row percent, count in brackets)")
    for i in range(N_DAMAGE):
        for j in range(N_DAMAGE):
            colour = "white" if percent[i, j] < 60 else "black"
            ax.text(j, i, f"{percent[i, j]:.1f}%\n({confusion[i, j]})", ha="center", va="center",
                    color=colour, fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="% of actual class")

    ax = axes[0, 1]
    bars = ax.bar(short_names, per_class, color="tab:orange")
    ax.axhline(overall, color="tab:red", linestyle="--", label=f"Overall accuracy {overall:.1f}%")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Diagnosis accuracy per damage type")
    for bar, value in zip(bars, per_class):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}%", ha="center")
    ax.legend(loc="lower right")

    ax = axes[1, 0]
    window = max(200, len(training_outcomes) // 100)
    if len(training_outcomes) >= window:
        curve = rolling_mean(training_outcomes, window) * 100.0
        ax.plot(np.arange(window, len(training_outcomes) + 1), curve, color="tab:blue")
    ax.set_xlabel("Training diagnoses")
    ax.set_ylabel(f"Rolling accuracy over {window} diagnoses (%)")
    ax.set_title("Learning curve")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    image = ax.imshow(mean_symptoms, cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(N_SENSORS))
    ax.set_xticklabels(SENSORS, rotation=30, ha="right")
    ax.set_yticks(range(N_DAMAGE))
    ax.set_yticklabels(DAMAGE_TYPES)
    ax.set_title("Mean observed symptom level per damage type")
    for i in range(N_DAMAGE):
        for j in range(N_SENSORS):
            ax.text(j, i, f"{mean_symptoms[i, j]:+.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="-2 severe low, +2 severe high")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_file, dpi=150)
    return overall, per_class, percent


def print_summary(overall, per_class, percent, train_steps, eval_steps):
    print(f"Training diagnoses: {train_steps:,}   Evaluation diagnoses (no learning): {eval_steps:,}")
    print("Normal operating ranges")
    for name, unit, low, high in zip(SENSORS, UNITS, NORMAL_LOW, NORMAL_HIGH):
        print(f"  {name:<12} {low:8.3f} to {high:8.3f} {unit}")
    print()
    print(f"Overall accuracy: {overall:.2f}%")
    for name, value in zip(DAMAGE_TYPES, per_class):
        print(f"  {name:<16} {value:6.2f}%")
    print()
    print("Most common misdiagnoses")
    mistakes = []
    for i in range(N_DAMAGE):
        for j in range(N_DAMAGE):
            if i != j:
                mistakes.append((percent[i, j], DAMAGE_TYPES[i], DAMAGE_TYPES[j]))
    mistakes.sort(reverse=True)
    for value, actual, guessed in mistakes[:5]:
        print(f"  {actual} mistaken for {guessed}: {value:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps", type=int, default=DEFAULT_TRAIN_STEPS)
    parser.add_argument("--eval-steps", type=int, default=DEFAULT_EVAL_STEPS)
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--eval-seed", type=int, default=2)
    parser.add_argument("--lr-decay", type=int, default=DEFAULT_LR_DECAY_STEPS)
    parser.add_argument("--out", default="lunar_diagnosis_report.png")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    env = LunarBaseEnv(n_robots=4)
    agent = SymptomAgent(lr_decay_steps=args.lr_decay)
    training_outcomes = run_training(env, agent, steps=args.train_steps, seed=args.train_seed)
    confusion, mean_symptoms = run_evaluation(env, agent, steps=args.eval_steps, seed=args.eval_seed)
    overall, per_class, percent = build_report(confusion, mean_symptoms, training_outcomes, args.out)
    print_summary(overall, per_class, percent, args.train_steps, args.eval_steps)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()