import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class BasicEnv(gym.Env):
    def __init__(self):
        super().__init__()
        import config.config as cfg
        self.cfg = cfg

        self.action_space = spaces.MultiDiscrete(
            [self.cfg.NUM_DIRECTIONS + self.cfg.num_offense_players] * self.cfg.num_offense_players
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.cfg.num_offense_players, 2 + 2 * (1 + self.cfg.num_offense_players + self.cfg.num_defense_players)),
                                            dtype=np.float32)

        self.reset()

    def get_angle_distance(self, from_, to_):
        angle = np.arctan2(to_[1] - from_[1], to_[0] - from_[0])
        distance = np.sqrt((to_[0] - from_[0]) ** 2 + (to_[1] - from_[1]) ** 2)
        return angle, distance

    def get_obs_state(self, ball_state, offense_state, defense_state):
        res = np.zeros((self.cfg.num_offense_players, 2 + 2 * (1 + self.cfg.num_offense_players + self.cfg.num_defense_players)))
        for i in range(self.cfg.num_offense_players):
            cnt = 2
            res[i, cnt:cnt + 2] = self.get_angle_distance(offense_state[i], ball_state[:2])
            cnt += 2
            for j in range(self.cfg.num_offense_players):
                res[i, cnt:cnt + 2] = self.get_angle_distance(offense_state[i], offense_state[j])
                cnt += 2
            for j in range(self.cfg.num_defense_players):
                res[i, cnt:cnt + 2] = self.get_angle_distance(offense_state[i], defense_state[j])
                cnt += 2

        offense_distances = res[:, 3]
        any_offense_has_ball = offense_distances.min() < self.cfg.offense_radius
        res[:, 1] = 1 if any_offense_has_ball else 0
        if any_offense_has_ball:
            res[np.argmin(offense_distances)] = 1

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

    def step(
            self,
            action  # array: (num offensive players, 8 moves + pass) of logits
    ):
        movements_array = np.array([
            [0.5 ** 0.5, 0.5 ** 0.5],
            [-0.5 ** 0.5, 0.5 ** 0.5],
            [0.5 ** 0.5, -0.5 ** 0.5],
            [-0.5 ** 0.5, -0.5 ** 0.5],
            [0, 1],
            [1, 0],
            [0, -1],
            [-1, 0],
        ]) * self.cfg.offsense_speed

        new_ball_state = np.zeros(self.ball_state.shape)
        new_offense_state = np.zeros(self.offense_state.shape)
        new_defense_state = np.zeros(self.defense_state.shape)

        any_had_ball = self.obs_state[0, 1]
        has_ball = self.obs_state[:, 0]
        ball_held = any_had_ball
        ball_velocity = self.ball_state[2:]
        defenders_win = False

        # defense moves first
        # hardcoded

        for i in range(self.cfg.num_defense_players):
            direction_difference = self.ball_state[:2] - self.defense_state[i]
            direction_difference /= (np.linalg.norm(direction_difference) + 1e-5)
            new_defense_state[i] = self.defense_state[i] + direction_difference * self.cfg.defense_speed
            if self.get_angle_distance(new_defense_state[i], self.ball_state[:2])[1] < self.cfg.defense_radius:
                defenders_win = True

        for i in range(self.cfg.num_offense_players):
            if action[i] < self.cfg.NUM_DIRECTIONS:
                new_offense_state[i] = self.offense_state[i] + movements_array[action[i]]
            else:
                if has_ball[i]:
                    to_player = action[i] - self.cfg.NUM_DIRECTIONS
                    if to_player == i:
                        continue
                    direction_difference = self.offense_state[to_player] - self.ball_state[:2]
                    direction_difference /= (np.linalg.norm(direction_difference) + 1e-5)
                    direction_difference * self.cfg.ball_speed
                    ball_velocity = direction_difference
                    ball_held = False

        if ball_held:
            player_with_ball = np.argmax(has_ball)
            new_ball_state[:2] = new_offense_state[player_with_ball]
        if not ball_held:
            new_ball_state[:2] = self.ball_state[:2] + ball_velocity
            new_ball_state[2:] = ball_velocity * self.cfg.ball_slowdown


        if defenders_win:
            reward = -self.cfg.board_width
            done = True
        else:
            reward = new_ball_state[0] - self.ball_state[0]
            done = self.step_count >= self.cfg.max_steps  # todo

        self.step_count += 1
        self.ball_state = new_ball_state
        self.defense_state = new_defense_state
        self.offense_state = new_offense_state

        info = {}
        return self.obs_state, reward, done, info

    def render(self, mode='human'):
        if mode == 'human':
            fig, ax = plt.subplots(figsize=(self.cfg.board_width / 3, self.cfg.board_height / 3))
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
