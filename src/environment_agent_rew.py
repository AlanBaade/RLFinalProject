import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class BasicEnv(gym.Env):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.action_space = spaces.MultiDiscrete(
            [self.cfg.NUM_DIRECTIONS + self.cfg.num_offense_players] * self.cfg.num_offense_players
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.cfg.num_offense_players, 2 + 2 * (1 + self.cfg.num_offense_players + self.cfg.num_defense_players) + 2),
                                            dtype=np.float32)

        self.reset()

        self.movements_array = np.array([
            [0.5 ** 0.5, 0.5 ** 0.5],
            [-0.5 ** 0.5, 0.5 ** 0.5],
            [0.5 ** 0.5, -0.5 ** 0.5],
            [-0.5 ** 0.5, -0.5 ** 0.5],
            [0, 1],
            [1, 0],
            [0, -1],
            [-1, 0],
        ])
        self.MOVE_RIGHT = 5

    def get_distance(self, from_, to_):
        distance = np.sqrt((to_[0] - from_[0]) ** 2 + (to_[1] - from_[1]) ** 2)
        return distance

    def get_obs_angle_distance(self, from_, to_):
        angle = np.arctan2(to_[1] - from_[1], to_[0] - from_[0])
        distance = self.get_distance(from_, to_) / self.cfg.board_height
        return angle, distance

    def get_obs_state(self, ball_state, offense_state, defense_state):
        res = np.zeros((self.cfg.num_offense_players, 2 + 2 * (1 + self.cfg.num_offense_players + self.cfg.num_defense_players) + 2))
        for i in range(self.cfg.num_offense_players):
            cnt = 2
            res[i, cnt:cnt + 2] = self.get_obs_angle_distance(offense_state[i], ball_state[:2])
            cnt += 2
            for j in range(self.cfg.num_offense_players):
                res[i, cnt:cnt + 2] = self.get_obs_angle_distance(offense_state[i], offense_state[j])
                cnt += 2
            for j in range(self.cfg.num_defense_players):
                res[i, cnt:cnt + 2] = self.get_obs_angle_distance(offense_state[i], defense_state[j])
                cnt += 2

        offense_distances = res[:, 3]
        any_offense_has_ball = offense_distances.min() * self.cfg.board_height < self.cfg.offense_radius
        res[:, 1] = 1 if any_offense_has_ball else 0
        if any_offense_has_ball:
            res[np.argmin(offense_distances)][0] = 1
        res[:, -2] = offense_state[:, 0] / self.cfg.board_width
        res[:, -1] = offense_state[:, 1] / self.cfg.board_height

        return res

    def constrain_bounds(self, ball_state, offense_state, defense_state):
        pass

    def reset(self, seed=0):
        self.offense_state = np.zeros((self.cfg.num_offense_players, 2))
        for i in range(self.cfg.num_offense_players):
            self.offense_state[i][0] = self.cfg.offense_start_x
            self.offense_state[i][1] = self.cfg.board_height * (i + 0.5) / self.cfg.num_offense_players

        self.defense_state = np.zeros((self.cfg.num_defense_players, 2))
        for i in range(self.cfg.num_defense_players):
            self.defense_state[i][0] = self.cfg.defense_start_x
            self.defense_state[i][1] = self.cfg.board_height * (i + 0.5) / self.cfg.num_defense_players

        self.offense_state += np.random.standard_normal(self.offense_state.shape) * self.cfg.random_start_noise
        self.defense_state += np.random.standard_normal(self.defense_state.shape) * self.cfg.random_start_noise

        self.ball_state = np.zeros((4,))
        self.ball_state[:2] = self.offense_state[self.cfg.num_offense_players // 2]

        self.step_count = 0
        self.obs_state = self.get_obs_state(self.ball_state, self.offense_state, self.defense_state)

        return self.obs_state, {}

    def agent_rew(self, offense_state_diff):
        reward = 0
        for i in range(self.cfg.num_offense_players):
            reward += offense_state_diff[i][0]
        reward /= float(self.cfg.num_offense_players)
        reward *= self.cfg.agent_progress_reward_scale
        return reward

    def step(
            self,
            action  # array: (num offensive players, 8 moves + pass) of logits
    ):
        new_ball_state = np.zeros(self.ball_state.shape)
        new_offense_state = np.zeros(self.offense_state.shape)
        new_defense_state = np.zeros(self.defense_state.shape)

        any_had_ball = self.obs_state[0, 1] == 1
        has_ball = self.obs_state[:, 0]
        ball_held = any_had_ball
        ball_velocity = self.ball_state[2:]
        defenders_win = False

        # defense moves first
        offense_ball_distances = self.obs_state[:, 3]
        defender_targets = np.argsort(offense_ball_distances)
        defense_target_points = np.zeros((self.cfg.num_defense_players, 2))
        defense_target_points[0] = self.ball_state[:2]
        i = 1 if ball_held else 0
        for j in range(1, self.cfg.num_defense_players):
            defense_target_points[j] = self.offense_state[defender_targets[i]]
            i += 1
        defense_target_points_aligned = np.zeros((self.cfg.num_defense_players, 2))
        defense_taken = set()
        for i in range(self.cfg.num_defense_players):
            min_defender = 0
            min_dist = np.inf
            for j in range(self.cfg.num_defense_players):
                if j in defense_taken:
                    continue
                cur_dist = self.get_distance(defense_target_points[i], self.defense_state[j])
                if cur_dist < min_dist:
                    min_defender = j
                    min_dist = cur_dist
            defense_taken.add(min_defender)
            defense_target_points_aligned[min_defender] = defense_target_points[i]

        for i in range(self.cfg.num_defense_players):
            direction_difference = defense_target_points_aligned[i] - self.defense_state[i]
            direction_difference /= (np.linalg.norm(direction_difference) + 1e-5)
            new_defense_state[i] = self.defense_state[i] + direction_difference * self.cfg.defense_speed
            if self.get_distance(new_defense_state[i], self.ball_state[:2]) < self.cfg.defense_radius:
                defenders_win = True

        for i in range(self.cfg.num_offense_players):
            if action[i] < self.cfg.NUM_DIRECTIONS:
                new_offense_state[i] = self.offense_state[i] + self.movements_array[action[i]] * self.cfg.offense_speed
            else:
                new_offense_state[i] = self.offense_state[i]
                if has_ball[i]:
                    to_player = action[i] - self.cfg.NUM_DIRECTIONS
                    if to_player == i:
                        continue
                    new_offense_state[i] = self.offense_state[i]
                    direction_difference = self.offense_state[to_player] - self.ball_state[:2]
                    direction_difference /= (np.linalg.norm(direction_difference) + 1e-5)
                    ball_velocity = direction_difference * self.cfg.ball_speed
                    ball_held = False

        if ball_held:
            player_with_ball = np.argmax(has_ball)
            new_ball_state[:2] = new_offense_state[player_with_ball]
        if not ball_held:
            new_ball_state[:2] = self.ball_state[:2] + ball_velocity
            new_ball_state[2:] = ball_velocity * self.cfg.ball_slowdown

        offenders_win = new_ball_state[0] >= self.cfg.board_width-1
        if defenders_win:
            reward = -self.cfg.board_width
            done = True
        elif offenders_win:
            reward = self.cfg.board_width - self.ball_state[0]
            done = True
        else:
            reward = new_ball_state[0] - self.ball_state[0] + self.agent_rew(new_offense_state - self.offense_state)
            done = self.step_count >= self.cfg.max_steps  # todo

        self.step_count += 1
        self.ball_state = new_ball_state
        self.offense_state = new_offense_state
        self.defense_state = new_defense_state

        self.ball_state[0:1] = self.ball_state[0:1].clip(min=0, max=self.cfg.board_width)
        self.ball_state[1:2] = self.ball_state[1:2].clip(min=0, max=self.cfg.board_height)
        self.offense_state[:, 0:1] = self.offense_state[:, 0:1].clip(min=0, max=self.cfg.board_width)
        self.offense_state[:, 1:2] = self.offense_state[:, 1:2].clip(min=0, max=self.cfg.board_height)
        self.defense_state[:, 0:1] = self.defense_state[:, 0:1].clip(min=0, max=self.cfg.board_width)
        self.defense_state[:, 1:2] = self.defense_state[:, 1:2].clip(min=0, max=self.cfg.board_height)

        self.obs_state = self.get_obs_state(self.ball_state, self.offense_state, self.defense_state)

        info = {}
        truncated = False
        return self.obs_state, reward, done, truncated, info

    def render(self, mode='human'):
        if mode == 'human':
            fig, ax = plt.subplots(figsize=(self.cfg.board_width / 10, self.cfg.board_height / 10))
            ax.set_xlim(-10, self.cfg.board_width)
            ax.set_ylim(0, self.cfg.board_height)
            for i in range(self.cfg.num_offense_players):
                ax.add_patch(plt.Circle(self.offense_state[i], self.cfg.offense_radius, facecolor='b', linewidth=1, edgecolor='k', linestyle='solid'))
            for i in range(self.cfg.num_defense_players):
                ax.add_patch(plt.Circle(self.defense_state[i], self.cfg.defense_radius, facecolor='r', linewidth=1, edgecolor='k', linestyle='solid'))
            ax.add_patch(plt.Circle(self.ball_state[:2], 0.5, facecolor='w', linewidth=1, edgecolor='k', linestyle='solid'))
            ax.set_facecolor('g')
            plt.show()

        return


gym.register(id='SoccerEnvAgentRew-v0', entry_point='src.environment_agent_rew:BasicEnv')
