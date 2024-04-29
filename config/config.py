# meta
is_cuda = False
reward_type = 'distance'

# env
board_width = 100
board_height = 30
max_steps = 100

num_offense_players = 3
num_defense_players = 2

offense_radius = 1.
defense_radius = 1.

offense_speed = 0.4
defense_speed = 0.5

NUM_DIRECTIONS = 8

offense_start_x = 0
defense_start_x = 10

ball_speed = 2.0
ball_slowdown = 0.9



assert ball_speed > offense_radius
assert defense_speed < defense_radius * 2
assert num_offense_players >= num_defense_players
assert ball_speed > defense_speed
assert defense_speed > offense_speed