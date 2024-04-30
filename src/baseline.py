import numpy as np


def baseline_policy(obs, env):
    action = np.zeros(env.cfg.num_offense_players, dtype=int)
    for i in range(env.cfg.num_offense_players):
        if obs[i][1] == 1:
            if obs[i][0] == 1:
                defender_distances = obs[i][4 + 2 * env.cfg.num_offense_players + 1:4 + 2 * (env.cfg.num_offense_players + env.cfg.num_defense_players) + 1:2]
                if defender_distances.min() * env.cfg.board_height < env.cfg.defense_speed * 2.5 + env.cfg.defense_radius * 2.0:
                    action[i] = env.cfg.NUM_DIRECTIONS + (i + 1) % env.cfg.num_offense_players
                else:
                    action[i] = env.MOVE_RIGHT
            else:
                if np.random.random() < 0.25:
                    action[i] = env.MOVE_RIGHT
                else:
                    action[i] = np.random.randint(0, env.cfg.NUM_DIRECTIONS)
        else:
            ball_angle = obs[i][2]
            action[i] = np.argmin(
                (
                        (np.array([[np.cos(ball_angle), np.sin(ball_angle)]]) - env.movements_array) ** 2
                ).sum(axis=1))
    return action
