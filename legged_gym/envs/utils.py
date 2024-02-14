# environments
from legged_gym.envs.go1.go1 import Go1
from legged_gym.envs.npc.go1_sheep import Go1Sheep
from legged_gym.envs.npc.go1_object import Go1Object

# configs
from legged_gym.envs.configs.go1_plane_config import Go1PlaneCfg
from legged_gym.envs.configs.go1_gate_config import Go1GateCfg
from legged_gym.envs.configs.go1_sheep_config import Go1SheepCfg
from legged_gym.envs.configs.go1_football_config import Go1FootballCfg
from legged_gym.envs.configs.go1_seesaw_config import Go1SeesawCfg
from legged_gym.envs.configs.go1_pushbox_config import Go1PushboxCfg

# wrappers
from legged_gym.envs.wrappers.empty_wrapper import EmptyWrapper
from legged_gym.envs.wrappers.go1_gate_wrapper import Go1GateWrapper
from legged_gym.envs.wrappers.go1_pushbox_wrapper import Go1PushboxWrapper
from legged_gym.envs.wrappers.go1_sheep_wrapper import Go1SheepWrapper
from legged_gym.envs.wrappers.go1_seesaw_wrapper import Go1SeesawWrapper
from legged_gym.envs.wrappers.go1_football_wrapper import Go1FootballWrapper

from legged_gym.utils import get_args, make_env

ENV_DICT = {
    "go1plane": {
        "class": Go1,
        "config": Go1PlaneCfg,
        "wrapper": EmptyWrapper
    },
    "go1gate": {
        "class": Go1,
        "config": Go1GateCfg,
        "wrapper": Go1GateWrapper
    },
    "go1sheep": {
        "class": Go1Sheep,
        "config": Go1SheepCfg,
        "wrapper": Go1SheepWrapper
    },
    "go1football": {
        "class": Go1Object,
        "config": Go1FootballCfg,
        "wrapper": Go1FootballWrapper
    },
    "go1seesaw": {
        "class": Go1Object,
        "config": Go1SeesawCfg,
        "wrapper": Go1SeesawWrapper
    },
    "go1pushbox": {
        "class": Go1Object,
        "config": Go1PushboxCfg,
        "wrapper": Go1PushboxWrapper
    },
}

def make_mqe_env(env_name, args=None, custom_cfg=None):
    
    env_dict = ENV_DICT[env_name]

    if callable(custom_cfg):
        env_dict["config"] = custom_cfg(env_dict["config"])

    env, env_cfg = make_env(env_dict["class"], env_dict["config"], args)
    env = env_dict["wrapper"](env)

    return env, env_cfg