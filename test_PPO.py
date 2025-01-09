import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from lwktoess import LWKToess

# Parameter
MODEL_PATH = "./worst_model/worst_model.zip"  # Pfad zum trainierten Modell
MODEL_PATH = "./best_model/best_model.zip"  # Pfad zum trainierten Modell

SIM_TIME = 600  # Simulationszeit pro Episode in Sekunden
DTIME = 0.2  # Zeitinkrement in der Umgebung
TIMESTEPS_PER_EPISODE = int(SIM_TIME / DTIME)  # Schritte pro Episode
GROUPS = {
    "Wasserstände": [0, 1],  # h_OT, h_NT
    "Leistungen": [2, 3, 4],  # Pel1, Pel2, Pel34
    "Leitwerke": [5, 6, 7, 8, 9, 10],  # alpha_1 bis alpha_6
    "Fehlerzeiten": [11, 12, 13, 14],  # FT1 bis FT4
}

def load_model_and_env(model_path, sim_time):
    """Lädt das Modell und initialisiert die Umgebung."""
    env = LWKToess(render_mode=None, sim_time=sim_time)
    model = PPO.load(model_path, env=env)
    return model, env

def run_single_episode(model, env):
    """Führt eine einzelne Episode aus und gibt die Beobachtungen zurück."""
    obs, _ = env.reset()
    observations = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        observations.append(obs)
    print(info["cumulated_reward"])
    return np.array(observations)

def plot_observations(all_observations, groups, dtime):
    """Plottet die Beobachtungen der Episode."""
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    time = np.arange(len(all_observations)) * dtime

    for ax, (group_name, indices) in zip(axes, groups.items()):
        for obs_idx in indices:
            ax.plot(
                time,
                all_observations[:, obs_idx],
                label=f"Observation {obs_idx}",
                alpha=0.7,
            )
        ax.set_title(group_name, fontsize=12)
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Zeit (s)", fontsize=12)
    plt.tight_layout()
    plt.show()

def main():
    """Hauptfunktion für das Testen des Modells."""
    model, env = load_model_and_env(MODEL_PATH, SIM_TIME)
    observations = run_single_episode(model, env)
    plot_observations(observations, GROUPS, DTIME)
    env.close()

if __name__ == "__main__":
    main()
