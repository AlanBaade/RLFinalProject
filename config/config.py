# meta
is_cuda = False
reward_type = 'distance'
random_start_noise = 2

# env
# board_width = 100
# board_height = 30
# max_steps = 200
# num_offense_players = 3
# num_defense_players = 2

board_width = 100
board_height = 75
max_steps = 200
num_offense_players = 11
num_defense_players = 6


offense_radius = 1.25
defense_radius = 1.3

offense_speed = 1.25
defense_speed = 1.5

NUM_DIRECTIONS = 8

offense_start_x = 0
defense_start_x = 20

ball_speed = 2.5
ball_slowdown = 0.95





assert ball_speed > offense_radius
assert defense_speed < defense_radius * 2
assert num_offense_players >= num_defense_players
assert ball_speed > defense_speed
assert defense_speed > offense_speed
