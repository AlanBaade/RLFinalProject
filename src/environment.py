import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class BasicEnv(gym.Env):
    def __init__(self):
        super().__init__()
        import config.config as cfg
        self.cfg = cfg

        self.action_space = spaces.Discrete(
            self.cfg.num_offense_players * (self.cfg.NUM_DIRECTIONS + self.cfg.num_offense_players - 1))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.cfg.num_offense_players, 1 + self.cfg.num_offense_players + self.cfg.num_defense_players, 2), dtype=np.float32)

        self.reset()

    def get_obs_state(self, ball_state, offense_state, defense_state):
        def get_angle_distance(from_x, from_y, to_x, to_y):
            angle = np.arctan2(to_y - from_y, to_x - from_x)
            distance = np.sqrt((to_x - from_x) ** 2 + (to_y - from_y) ** 2)
            return angle, distance

        res = np.zeros((self.cfg.num_offense_players, 1 + self.cfg.num_offense_players + self.cfg.num_defense_players, 2))
        for i in range(self.cfg.num_offense_players):
            cnt = 0
            res[i, cnt] = get_angle_distance(*offense_state[i], *ball_state[:2])
            cnt += 1
            for j in range(self.cfg.num_offense_players):
                if i == j:
                    continue
                res[i, cnt] = get_angle_distance(*offense_state[i], *offense_state[j])
                cnt += 1
            for j in range(self.cfg.num_defense_players):
                res[i, cnt] = get_angle_distance(*offense_state[i], *defense_state[j])
                cnt += 1

        return res

    def constrain_bounds(self, ball_state, offense_state, defense_state):
        pass

    def reset(self):
        self.offense_state = np.zeros((self.cfg.num_offense_players, 2))
        for i in range(self.cfg.num_offense_players):
            self.offense_state[i][0] = self.cfg.offense_start_x
            self.offense_state[i][1] = self.cfg.board_height * (i + 0.5) / self.cfg.num_offense_players

        self.defense_state = np.zeros((self.cfg.num_defense_players, 2))
        for i in range(self.cfg.num_defense_players):
            self.defense_state[i][0] = self.cfg.defense_start_x
            self.defense_state[i][1] = self.cfg.board_height * (i + 0.5) / self.cfg.num_defense_players

        self.ball_state = np.zeros((4,))
        self.ball_state[:2] = self.offense_state[self.cfg.num_offense_players // 2]

        self.step_count = 0
        self.obs_state = self.get_obs_state(self.ball_state, self.offense_state, self.defense_state)
        print(self.obs_state)

        return self.obs_state, {}

    def step(self, action):
        self.step_count += 1
        reward = 0
        done = self.step_count >= self.cfg.max_steps  # todo
        info = {}

        return self.obs_state, reward, done, info

    def render(self, mode='human'):
        if mode == 'human':
            fig, ax = plt.subplots(figsize=(self.cfg.board_width/3, self.cfg.board_height/3))
            ax.set_xlim(-10, self.cfg.board_width)
            ax.set_ylim(0, self.cfg.board_height)
            for i in range(self.cfg.num_offense_players):
                ax.add_patch(plt.Circle(self.offense_state[i], 1, facecolor='b', linewidth=1, edgecolor='k', linestyle='solid'))
            for i in range(self.cfg.num_defense_players):
                ax.add_patch(plt.Circle(self.defense_state[i], 1, facecolor='r', linewidth=1, edgecolor='k', linestyle='solid'))
            ax.add_patch(plt.Circle(self.ball_state[:2], 0.5, facecolor='w', linewidth=1, edgecolor='k', linestyle='solid'))
            ax.set_facecolor('g')
            plt.show()

        return


gym.register(id='SoccerEnv-v0', entry_point='src.environment:BasicEnv')
