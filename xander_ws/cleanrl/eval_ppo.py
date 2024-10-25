import os
import time
from dataclasses import dataclass
from datetime import datetime
import isaacgym
import torch
import torch.nn as nn
import gym
import numpy as np
from mqe.utils import get_args
from mqe.envs.utils import make_mqe_env, custom_cfg

# 定义按键监听器函数
def keyboard_listener(exit_flag):
    import sys
    import select
    while True:
        dr, dw, de = select.select([sys.stdin], [], [], 0)
        if dr:
            key = sys.stdin.readline()
            if key.strip().lower() == 'q':
                exit_flag.append(True)
                break
        time.sleep(0.1)

# 定义Agent类，与训练时的结构相同
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, np.prod(envs.single_action_space.shape)),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_action(self, x):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        return action

# 定义评估函数
def evaluate_model(model_path, device):
    # 设置环境参数
    task_name = "go1football-shoot"
    env_args = get_args()
    env_args.num_envs = 1  # 设置为1，方便可视化
    # env_args.rl_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # env_args.sim_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_args.headless = False  # 设置为False，启用渲染
    env_args.record_video = False
    envs, _ = make_mqe_env(task_name, env_args, custom_cfg(env_args))

    # 创建agent并加载模型
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(model_path))

    # 评估agent
    obs = envs.reset()
    total_reward = 0
    done = [False]
    while True:
        if exit_flag:
            break
        
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        with torch.no_grad():
            action = agent.get_action(obs_tensor)
        # action = action.cpu().numpy()
        obs, reward, done, info = envs.step(action)
        total_reward += reward
        # envs.render(mode='human')  # 渲染环境
        # time.sleep(0.02)  # 控制播放速度
        

    print(f"Episode finished. Total reward: {total_reward[0]}")
    envs.close()

if __name__ == "__main__":
    import threading
    exit_flag = []
    listener_thread = threading.Thread(target=keyboard_listener, args=(exit_flag,))
    listener_thread.daemon = True
    listener_thread.start()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "path/to/your/trained_model.pth"  # 请替换为您实际的模型路径
    evaluate_model(model_path, device)
