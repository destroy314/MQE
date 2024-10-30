import numpy as np
import os
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

if __name__ == '__main__':
    args = get_args()
    task_name = "go1football-shoot"
    args.task = task_name
    args.num_envs = 2048
    args.seed = 0
    args.compute_device_id = 0
    args.graphics_device_id = 0
    args.rl_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.sim_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.headless = True
    args.record_video = False
    
    env, env_cfg = make_mqe_env(task_name, args, custom_cfg(args))

    train_cfg = LeggedRobotCfgPPO()
    _, train_cfg = update_cfg_from_args(None, train_cfg, args)
    train_cfg_dict = class_to_dict(train_cfg)
        
    log_root = "default"
    if log_root == "default":
        log_root = os.path.join(RSLRL_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
        log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
    elif log_root is None:
        log_dir = None
    else:
        log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)

    runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
    
    # save resume path before creating a new log_dir
    resume = train_cfg.runner.resume
    if resume:
        # load previously trained model
        resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
        print(f"Loading model from: {resume_path}")
        runner.load(resume_path)
    
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)