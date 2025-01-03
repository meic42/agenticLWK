from gymnasium.utils.play import play
from GymEnv.LWKToess import LWKToess

env = LWKToess(render_mode='rgb_array', sim_time=600)

play(
    env,
    keys_to_action={
        "w": 1,
        "s": 2,
        "e": 3,
        "d": 4,
        "r": 5,
        "f": 6,
        "u": 7,
        "j": 8,
        "i": 9,
        "k": 10,
        "o": 11,
        "l": 12       
    },
    noop=0
)