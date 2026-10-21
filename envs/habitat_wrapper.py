import numpy as np
import torch
import gym
from gym import spaces

try:
    import habitat
    from habitat import Config
    HAS_HABITAT = True
except ImportError:
    HAS_HABITAT = False


class ImageNavEnv(gym.Env):
    """
    Gym-compatible wrapper around Habitat's ImageGoalNav task.

    Provides:
        - obs:  dict with 'rgb' (128,128,3) uint8 and 'goal' (128,128,3) uint8
        - reward: RSRNav paper reward (distance reduction + angle reduction + terminal)
        - info:  dict with 'distance_to_goal', 'success', 'spl', etc.

    Action space: Discrete(4)
        0: MOVE_FORWARD  (0.25m)
        1: TURN_LEFT     (30°)
        2: TURN_RIGHT    (30°)
        3: STOP
    """

    ACTION_MAP = {
        0: "MOVE_FORWARD",
        1: "TURN_LEFT",
        2: "TURN_RIGHT",
        3: "STOP",
    }

    def __init__(self, config_path=None, scene_ids=None, max_steps=500,
                 step_penalty=0.01, success_reward=5.0, success_distance=1.0):
        super().__init__()

        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.success_reward = success_reward
        self.success_distance = success_distance

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),
            "goal": spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8),
        })

        self._hab_env = None
        self._config_path = config_path
        self._scene_ids = scene_ids
        self._step_count = 0
        self._prev_distance = None
        self._prev_angle = None
        self._goal_image = None
        self._episode_distance = None

        if HAS_HABITAT and config_path is not None:
            self._init_habitat(config_path, scene_ids)

    def _init_habitat(self, config_path, scene_ids=None):
        """Initialize the underlying Habitat environment."""
        config = habitat.get_config(config_path)

        if scene_ids is not None:
            with habitat.config.read_write(config):
                config.habitat.dataset.content_scenes = scene_ids

        self._hab_env = habitat.Env(config=config)

    def _get_obs(self, hab_obs):
        """Extract RGB observation and goal image from Habitat obs dict."""
        rgb = hab_obs.get("rgb", np.zeros((128, 128, 3), dtype=np.uint8))
        goal = self._goal_image
        return {"rgb": rgb, "goal": goal}

    def _compute_reward(self, hab_info, action):
        """
        RSRNav reward function (Section III-D):
            r_step = r_d + [d_t < d_{t-1}] * r_α - γ
            r_terminal = 5 (if success) + 5 (if angle < threshold)
        """
        curr_distance = hab_info.get("distance_to_goal", 0.0)
        curr_angle = hab_info.get("angle_to_goal", 0.0)

        # Distance reduction reward
        r_d = 0.0
        if self._prev_distance is not None:
            r_d = self._prev_distance - curr_distance

        # Angle reduction reward (only when getting closer)
        r_a = 0.0
        if self._prev_angle is not None and r_d > 0:
            r_a = self._prev_angle - curr_angle

        reward = r_d + r_a - self.step_penalty

        # Terminal reward
        done = False
        success = False
        if action == 3:  # STOP
            done = True
            if curr_distance < self.success_distance:
                success = True
                reward += self.success_reward
                if abs(curr_angle) < np.radians(30):
                    reward += self.success_reward

        if self._step_count >= self.max_steps:
            done = True

        self._prev_distance = curr_distance
        self._prev_angle = curr_angle

        return reward, done, success

    def reset(self):
        if self._hab_env is not None:
            hab_obs = self._hab_env.reset()
            self._goal_image = hab_obs.get("imagegoal",
                                           np.zeros((128, 128, 3), dtype=np.uint8))
            metrics = self._hab_env.get_metrics()
            self._prev_distance = metrics.get("distance_to_goal", float("inf"))
            self._prev_angle = metrics.get("angle_to_goal", 0.0)
            self._episode_distance = self._prev_distance
            self._step_count = 0
            return self._get_obs(hab_obs)
        else:
            return self._dummy_reset()

    def step(self, action):
        self._step_count += 1

        if self._hab_env is not None:
            hab_action = self._to_habitat_action(action)
            hab_obs = self._hab_env.step(hab_action)
            hab_info = self._hab_env.get_metrics()
            reward, done, success = self._compute_reward(hab_info, action)

            obs = self._get_obs(hab_obs)
            info = {
                "distance_to_goal": hab_info.get("distance_to_goal", 0.0),
                "success": success,
                "spl": hab_info.get("spl", 0.0),
                "num_steps": self._step_count,
                "episode_distance": self._episode_distance,
            }
            return obs, reward, done, info
        else:
            return self._dummy_step(action)

    def _to_habitat_action(self, action):
        """Convert integer action to Habitat action."""
        action_name = self.ACTION_MAP[action]
        return self._hab_env.task.action_space[action_name]

    def close(self):
        if self._hab_env is not None:
            self._hab_env.close()

    # --- Dummy methods for testing without Habitat installed ---

    def _dummy_reset(self):
        self._step_count = 0
        self._prev_distance = np.random.uniform(3, 15)
        self._prev_angle = np.random.uniform(-np.pi, np.pi)
        self._episode_distance = self._prev_distance
        self._goal_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        return {
            "rgb": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
            "goal": self._goal_image,
        }

    def _dummy_step(self, action):
        self._step_count += 1
        new_dist = max(0, self._prev_distance + np.random.uniform(-0.5, 0.3))
        new_angle = self._prev_angle + np.random.uniform(-0.3, 0.3)

        info_dict = {"distance_to_goal": new_dist, "angle_to_goal": new_angle}
        reward, done, success = self._compute_reward(info_dict, action)

        obs = {
            "rgb": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
            "goal": self._goal_image,
        }
        info = {
            "distance_to_goal": new_dist,
            "success": success,
            "spl": 0.0,
            "num_steps": self._step_count,
            "episode_distance": self._episode_distance,
        }
        return obs, reward, done, info


class VectorizedImageNavEnv:
    """
    Simple vectorized wrapper that runs N ImageNavEnv instances in sequence.
    (Use multiprocessing SubprocVecEnv for actual training.)
    """

    def __init__(self, num_envs, **env_kwargs):
        self.envs = [ImageNavEnv(**env_kwargs) for _ in range(num_envs)]
        self.num_envs = num_envs

    def reset(self):
        obs_list = [env.reset() for env in self.envs]
        return self._stack_obs(obs_list)

    def step(self, actions):
        results = []
        for env, a in zip(self.envs, actions):
            obs, reward, done, info = env.step(a)
            if done:
                obs = env.reset()
            results.append((obs, reward, done, info))

        obs_list = [r[0] for r in results]
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        dones = np.array([r[2] for r in results], dtype=bool)
        infos = [r[3] for r in results]

        return self._stack_obs(obs_list), rewards, dones, infos

    def _stack_obs(self, obs_list):
        return {
            "rgb": np.stack([o["rgb"] for o in obs_list]),
            "goal": np.stack([o["goal"] for o in obs_list]),
        }

    def close(self):
        for env in self.envs:
            env.close()
