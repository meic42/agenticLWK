from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from GymEnv.LWKToess import LWKToess

# 1. RewardLoggerCallback für kumulierten Reward zu TensorBoard
class RewardLoggerCallback(BaseCallback):
    """
    Custom Callback for logging the cumulative reward to TensorBoard.
    """
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Infos aus der Umgebung abrufen
        infos = self.locals["infos"]
        for info in infos:
            if "cumulated_reward" in info:  # Wenn cumulated_reward in info vorhanden
                cumulated_reward = info["cumulated_reward"]
                self.logger.record("custom/cumulated_reward", cumulated_reward)
        return True

# 2. EvalCallback für das Speichern des besten Modells
eval_env = LWKToess(render_mode=None, sim_time=600)  # Separate Umgebung für Evaluierung
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./eval_logs/",
    eval_freq=10000,  # Evaluierung alle 10.000 Schritte
    n_eval_episodes=5,  # Über 5 Episoden evaluieren
    deterministic=True,
    render=False
)

# 3. Optional: Callback zum Speichern des schlechtesten Modells
class SaveWorstModelCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super(SaveWorstModelCallback, self).__init__(verbose)
        self.worst_reward = float("inf")
        self.save_path = save_path

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for info in infos:
            if "cumulated_reward" in info:
                cumulated_reward = info["cumulated_reward"]
                if cumulated_reward < self.worst_reward:
                    self.worst_reward = cumulated_reward
                    self.model.save(f"{self.save_path}/worst_model.zip")
                    if self.verbose > 0:
                        print(f"Schlechtestes Modell gespeichert mit Reward: {cumulated_reward}")
        return True

worst_model_callback = SaveWorstModelCallback(save_path="./worst_model/", verbose=1)

# Erstelle die Trainingsumgebung
env = make_vec_env(lambda: LWKToess(render_mode=None, sim_time=600), n_envs=8)

# Initialisiere das Modell
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_lwktoess_tensorboard/"
)

# Trainiere das Modell
model.learn(
    total_timesteps=10_000_000,  # Gesamtanzahl der Trainingsschritte
    callback=[RewardLoggerCallback(), eval_callback, worst_model_callback]  # Alle Callbacks einbinden
)

# Schließe die Umgebung
env.close()
