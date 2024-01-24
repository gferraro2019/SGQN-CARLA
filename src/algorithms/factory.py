from algorithms.curl import CURL
from algorithms.drq import DrQ
from algorithms.pad import PAD
from algorithms.rad import RAD
from algorithms.sac import SAC
from algorithms.sgsac import SGSAC
from algorithms.soda import SODA
from algorithms.svea import SVEA

algorithm = {
    "sac": SAC,
    "rad": RAD,
    "curl": CURL,
    "pad": PAD,
    "soda": SODA,
    "drq": DrQ,
    "svea": SVEA,
    "sgsac": SGSAC,
}


def make_agent(obs_shape, action_shape,env_action_spaces, args):
    return algorithm[args.algorithm](obs_shape, action_shape,env_action_spaces, args)
