from __future__ import annotations

import numpy as np
import os, re
from datetime import datetime
import isaacgym
from mqe.utils import get_args
from mqe.envs.utils import make_mqe_env, custom_cfg
from mqe.utils.helpers import update_cfg_from_args, class_to_dict, get_load_path
import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
import sys
sys.path.append("..")
from rslrl.config.train_cfg import LeggedRobotCfgPPO
from rslrl import RSLRL_ROOT_DIR
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, EmpiricalNormalization

from tqdm import tqdm


def get_latest_model_file(directory):
    pattern = re.compile(r'model_(\d+)\.pt')
    max_number = -1
    latest_file = None
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number
                latest_file = filename
    return latest_file


if __name__ == '__main__':
    args = get_args()
    task_name = "go1football-shoot"
    args.task = task_name
    args.num_envs = 1
    args.seed = 42
    args.compute_device_id = 0
    args.graphics_device_id = 0
    args.rl_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.sim_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.headless = False
    args.record_video = False
    
    env, env_cfg = make_mqe_env(task_name, args, custom_cfg(args))

    train_cfg = LeggedRobotCfgPPO()
    _, train_cfg = update_cfg_from_args(None, train_cfg, args)
    train_cfg_dict = class_to_dict(train_cfg)
    
    # load policy
    experiment_name = 'test'
    experiment_time = 'Oct29_22-56-48_'
    run_name = ''
    policy_root = os.path.join(RSLRL_ROOT_DIR, 'logs', experiment_name)
    policy_dir = os.path.join(policy_root, experiment_time + run_name)
    policy_path = os.path.join(policy_dir, get_latest_model_file(policy_dir))
    loaded_dict = torch.load(policy_path)
    print(f"iter: {loaded_dict['iter']}")

    # actor_critic
    policy_cfg = train_cfg_dict["policy"]
    num_obs = env.get_observations().shape[1]
    num_critic_obs = num_obs
    actor_critic_class = eval(policy_cfg.pop("class_name"))
    actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            num_obs, num_critic_obs, env.action_space.shape[0], **policy_cfg).to(args.rl_device)
    actor_critic.load_state_dict(loaded_dict["model_state_dict"])
    # alg_cfg = train_cfg_dict["algorithm"]
    # alg_class = eval(alg_cfg.pop("class_name"))  # PPO
    # alg: PPO = alg_class(actor_critic, device=args.device, **alg_cfg)
    
    # empirical normalization
    empirical_normalization = train_cfg_dict["runner"]["empirical_normalization"]
    if empirical_normalization:
        obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(args.rl_device)
        obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
    else:
        obs_normalizer = torch.nn.Identity().to(args.rl_device)  # no normalization
        
    # eval
    actor_critic.eval()
    if empirical_normalization:
        obs_normalizer.eval()
        
    num_eval_steps = 5000
    obs = env.reset()
    with torch.inference_mode():
        for i in tqdm(range(num_eval_steps)):
            actions = actor_critic.act_inference(obs)
            obs, rew, _, _ = env.step(actions)
            obs = obs_normalizer(obs)
        