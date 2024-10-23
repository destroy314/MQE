import gym
from gym import spaces
import numpy
import torch
from copy import copy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper
from isaacgym.torch_utils import *

# defender, game, shoot

class Go1FootballDefenderWrapper(EmptyWrapper):
    '''
    num_agents: 2
    Obs: shape (20,)
    Action: shape (3,)

    return:
        obs: (num_envs, num_agents, obs_dim)
        reward: (num_envs, num_agents)
        termination: (num_envs) 
        info: episode: max_pos_x, max_pos_y, min_pos_x, min_pos_y, pos_x, rew_ball_gate_distance_reward_scale, rew_frame_ball_gate_distance_reward_scale, rew_frame_goal_reward_scale, rew_goal_reward_scale
              time_outs: tensor.bool
    '''
    def __init__(self, env):
        super().__init__(env)

        self.num_agents = 2

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(18 + self.num_agents,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device="cuda").repeat(self.num_envs, 2, 1)

        self.obs_ids = torch.eye(self.num_agents, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1).reshape(self.num_envs, self.num_agents, -1)

        # for hard setting of reward scales (not recommended)
        
        # self.target_reward_scale = 1
        # self.success_reward_scale = 0
        # self.lin_vel_x_reward_scale = 0
        # self.approach_frame_punishment_scale = 0
        # self.agent_distance_punishment_scale = 0
        # self.lin_vel_y_punishment_scale = 0
        # self.command_value_punishment_scale = 0

        self.reward_buffer = {
            "goal reward": 0,
            "ball gate distance reward": 0,
            "step count": 0
        }

    def _init_extras(self, obs):
        return

    def reset(self):
        obs_buf = self.env.reset()

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)

        ball_pos = self.root_states_npc[:, :3].reshape(self.num_envs, 3) - self.env_origins
        ball_pos = ball_pos.unsqueeze(1).repeat(1, 2, 1)

        ball_vel = self.root_states_npc[:, 7:10].reshape(self.num_envs, 3).unsqueeze(1).repeat(1, 2, 1)

        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])[:, :2, :]
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]), ball_pos, ball_vel], dim=2)

        return obs

    def step(self, action):
        '''
        action: shape (num_envs, num_agents, 3)
        '''
        action = torch.clip(action, -1, 1)
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)
        
        ball_pos = self.root_states_npc[:, :3].reshape(self.num_envs, 3) - self.env_origins
        ball_pos = ball_pos.unsqueeze(1).repeat(1, 2, 1)

        ball_vel = self.root_states_npc[:, 7:10].reshape(self.num_envs, 3).unsqueeze(1).repeat(1, 2, 1)

        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])[:, :2, :]
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]), ball_pos, ball_vel], dim=2)

        self.reward_buffer["step count"] += 1
        reward = torch.zeros([self.env.num_envs, 1], device=self.env.device)

        if self.goal_reward_scale != 0:

            goal_reward = reward.clone()
            goal_reward[ball_pos[:, 0, 0] > self.gate_pos[:, 0], 0] = self.goal_reward_scale
            reward += goal_reward
            self.reward_buffer["goal reward"] += torch.sum(goal_reward).cpu()

        if self.ball_gate_distance_reward_scale != 0:
            
            ball_gate_distance = torch.norm(ball_pos[:, 0, :2] - self.gate_pos[:, :2], dim=1, keepdim=True)
            ball_gate_distance_reward = self.ball_gate_distance_reward_scale * torch.exp(-ball_gate_distance / 3)
            reward += ball_gate_distance_reward
            self.reward_buffer["ball gate distance reward"] += torch.sum(ball_gate_distance_reward).cpu()

        return obs, reward.repeat(1, 2), termination, info
    
class Go1FootballGameWrapper(EmptyWrapper):
    '''
    go1football-2vs2:
        num_agents: 4
        Obs: shape (22,)
        Action: shape (3,)
    go1football-1vs1:
        num_agents: 2
        Obs: shape (20,)
        Action: shape (3,)
    '''
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(18 + self.num_agents,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device="cuda").repeat(self.num_envs, self.num_agents, 1)

        self.obs_ids = torch.eye(self.num_agents, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1).reshape(self.num_envs, self.num_agents, -1)

        # for hard setting of reward scales (not recommended)
        
        # self.target_reward_scale = 1
        # self.success_reward_scale = 0
        # self.lin_vel_x_reward_scale = 0
        # self.approach_frame_punishment_scale = 0
        # self.agent_distance_punishment_scale = 0
        # self.lin_vel_y_punishment_scale = 0
        # self.command_value_punishment_scale = 0

        self.reward_buffer = {
            "goal reward": 0,
            "step count": 0
        }

    def _init_extras(self, obs):
        self.gate_pos = obs.env_info["gate_deviation"]
        self.gate_pos[:, 0] += self.BarrierTrack_kwargs["init"]["block_length"] + self.BarrierTrack_kwargs["gate"]["block_length"] / 2
        # self.gate_pos = self.gate_pos.unsqueeze(1).repeat(1, self.num_agents, 1)
        self.gate_distance = self.gate_pos.reshape(-1, 2)[:, 0]

    def reset(self):
        obs_buf = self.env.reset()

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)

        ball_pos = self.root_states_npc[:, :3].reshape(self.num_envs, 3) - self.env_origins  # (num_envs, 3)
        # ball_pos = ball_pos.unsqueeze(1).repeat(1, 2, 1)   # (num_envs, num_agents, 3)

        # ball_vel = self.root_states_npc[:, 7:10].reshape(self.num_envs, 3).unsqueeze(1).repeat(1, 2, 1)  # (num_envs, num_agents, 3)
        ball_vel = self.root_states_npc[:, 7:10].reshape(self.num_envs, 3)  # (num_envs, 3)

        base_pos = obs_buf.base_pos  # (num_envs*num_agents, 3) 奇数时为agent1, 偶数时为agent2
        base_rpy = obs_buf.base_rpy  # (num_envs*num_agents, 3) 奇数时为agent1, 偶数时为agent2
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])[:, :2, :]  # (num_envs, num_agents, 6)
        obs = torch.cat([ball_pos, ball_vel, base_pos.reshape(self.num_envs, -1), base_rpy.reshape(self.num_envs, -1), self.gate_pos], dim=1)
        # (num_envs, 18) 缺少门的位置信息
        return obs

    def step(self, action):
        action = torch.clip(action, -1, 1)
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)
        
        ball_pos = self.root_states_npc[:, :3].reshape(self.num_envs, 3) - self.env_origins
        # ball_pos = ball_pos.unsqueeze(1).repeat(1, 2, 1)

        # ball_vel = self.root_states_npc[:, 7:10].reshape(self.num_envs, 3).unsqueeze(1).repeat(1, 2, 1)
        ball_vel = self.root_states_npc[:, 7:10].reshape(self.num_envs, 3)

        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])[:, :2, :]

        obs = torch.cat([ball_pos, ball_vel, base_pos.reshape(self.num_envs, -1), base_rpy.reshape(self.num_envs, -1)], dim=1)
        # (num_envs, 18) 缺少门的位置信息

        self.reward_buffer["step count"] += 1
        reward = torch.zeros([self.env.num_envs, 1], device=self.env.device)



        return obs, reward.repeat(1, self.num_agents), termination, info
    

class Go1FootballShootWrapper(EmptyWrapper):
    '''
    go1football-shoot:
        num_agents: 1
        Obs: shape ()
        Action: shape (3,)
    '''
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(18 + self.num_agents,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device="cuda").repeat(self.num_envs, self.num_agents, 1)

        self.obs_ids = torch.eye(self.num_agents, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1).reshape(self.num_envs, self.num_agents, -1)

        # for hard setting of reward scales (not recommended)
        
        # self.target_reward_scale = 1
        # self.success_reward_scale = 0
        # self.lin_vel_x_reward_scale = 0
        # self.approach_frame_punishment_scale = 0
        # self.agent_distance_punishment_scale = 0
        # self.lin_vel_y_punishment_scale = 0
        # self.command_value_punishment_scale = 0

        self.reward_buffer = {
            "goal reward": 0,
            "step count": 0
        }
        

    def _init_extras(self, obs):
        self.gate_pos = obs.env_info["gate_deviation"]
        self.gate_pos[:, 0] += self.BarrierTrack_kwargs["init"]["block_length"] + self.BarrierTrack_kwargs["gate"]["block_length"] / 2
        # self.gate_pos = self.gate_pos.unsqueeze(1).repeat(1, self.num_agents, 1)
        self.gate_distance = self.gate_pos.reshape(-1, 2)[:, 0]

    def reset(self):
        obs_buf = self.env.reset()

        if getattr(self, "gate_pos", None) is None:
            # self._init_extras(obs_buf)
            self.gate_pos = torch.zeros([self.num_envs, 3], device=self.device)
            self.gate_pos[:, 0] = 12.0
            self.gate_pos[:, 1] = 0.0
            self.gate_pos[:, 2] = 0.0
        
        base_pos = obs_buf.base_pos
        base_quat = obs_buf.base_quat
        base_lin_vel = obs_buf.lin_vel
        base_ang_vel = obs_buf.ang_vel
        
        # self.root_states_npc: position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13])
        ball_pos = self.root_states_npc[:, :3].reshape(self.num_envs, 3) - self.env_origins
        ball_vel = self.root_states_npc[:, 7:10].reshape(self.num_envs, 3)
        
        base_lin_vel_2d = base_lin_vel[:, :2]
        base_vel_yaw = base_ang_vel[:, 2].unsqueeze(1)
        
        ball_pos_ego = quat_rotate(quat_conjugate(base_quat) , ball_pos - base_pos)[..., :2]
        ball_vel_ego = quat_rotate(quat_conjugate(base_quat) , ball_vel - base_lin_vel)[..., :2]
        goal_pos_ego = quat_rotate(quat_conjugate(base_quat) , self.gate_pos - base_pos)[..., :2]
        
        obs = torch.cat([base_lin_vel_2d, base_vel_yaw, ball_pos_ego, ball_vel_ego, goal_pos_ego], dim=1)   # (num_envs, 9) 
        
        return obs

    def step(self, action):
        """
        1. obs(以自身为坐标原点):
            self_lin_vel: 2 dims
            self_vel_yaw: 1 dim
            ball_pos: 2 dims
            ball_vel: 2 dims (需要转换为相对于自身的速度)
            goal_pos: 2 dims
        
        2. reward(参考dribblebot):
            reward_goal
            reward_dribbling_robot_ball_vel: encourage robot velocity align vector from robot body to ball
            reward_dribbling_robot_ball_pos: encourage robot near ball
            reward_dribbling_ball_vel: encourage ball vel align with unit vector between ball target(goal) and ball current position
            reward_dribbling_robot_ball_yaw: 鼓励机器狗的朝向与球的朝向一致
            
        """
        action = torch.clip(action, -1, 1)
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))
        
        # if getattr(self, "gate_pos", None) is None:
        #     self._init_extras(obs_buf)
        
        # obs
        base_pos = obs_buf.base_pos
        base_quat = obs_buf.base_quat
        base_lin_vel = obs_buf.lin_vel
        base_ang_vel = obs_buf.ang_vel
        
        # self.root_states_npc: position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13])
        ball_pos = self.root_states_npc[:, :3].reshape(self.num_envs, 3) - self.env_origins
        ball_vel = self.root_states_npc[:, 7:10].reshape(self.num_envs, 3)
        
        base_lin_vel_2d = base_lin_vel[:, :2]
        base_vel_yaw = base_ang_vel[:, 2].unsqueeze(1)
        
        ball_pos_ego = quat_rotate(quat_conjugate(base_quat) , ball_pos - base_pos)[..., :2]
        ball_vel_ego = quat_rotate(quat_conjugate(base_quat) , ball_vel - base_lin_vel)[..., :2]
        goal_pos_ego = quat_rotate(quat_conjugate(base_quat) , self.gate_pos - base_pos)[..., :2]
        
        obs = torch.cat([base_lin_vel_2d, base_vel_yaw, ball_pos_ego, ball_vel_ego, goal_pos_ego], dim=1)
        
        # reward
        self.reward_buffer["step count"] += 1
        
        reward = torch.zeros([self.env.num_envs, 1], device=self.device)
                
        # reward_goal
        _rew_goal = torch.where(ball_pos[:, 0] > self.gate_pos[:, 0], 1.0, 0.0).unsqueeze(1)
        
        # reward_robot_ball_goal: 鼓励机器狗，球，球门三者在一条直线上
        robot_ball_vec = (ball_pos - base_pos)[..., :2]
        d_robot_ball = robot_ball_vec / torch.norm(robot_ball_vec, dim=-1, keepdim=True)
        d_robot_ball = torch.nan_to_num(d_robot_ball, nan=0.0)
        ball_goal_vec = (self.gate_pos - ball_pos)[..., :2]
        d_ball_goal = ball_goal_vec / torch.norm(ball_goal_vec, dim=-1, keepdim=True)
        d_ball_goal = torch.nan_to_num(d_ball_goal, nan=0.0)
        
        robot_ball_goal_error = torch.norm(d_robot_ball, dim=-1) - torch.sum(d_robot_ball * d_ball_goal, dim=-1)
        
        delta_robot_ball_goal = 2.0
        _rew_robot_ball_goal = torch.exp(-delta_robot_ball_goal * robot_ball_goal_error)
        
        # reward_dribbling_robot_ball_vel: encourage robot velocity align vector from robot body to ball
        d_base_lin_vel_2d = base_lin_vel_2d / torch.norm(base_lin_vel_2d, dim=-1, keepdim=True)
        d_base_lin_vel_2d = torch.nan_to_num(d_base_lin_vel_2d, nan=0.0)
        robot_ball_vel_error = torch.norm(d_base_lin_vel_2d, dim=-1) - torch.sum(d_base_lin_vel_2d * d_robot_ball, dim=-1)
        
        delta_robot_ball_vel = 2.0
        _rew_robot_ball_vel = torch.exp(-delta_robot_ball_vel * robot_ball_vel_error)
        
        # reward_dribbling_robot_ball_pos: encourage robot near ball
        _rew_robot_ball_pos = torch.norm(robot_ball_vec, dim=-1, keepdim=True)
        
        # reward_ball_to_goal: encourage ball near goal
        _rew_ball_to_goal = torch.norm(ball_goal_vec, dim=-1, keepdim=True)
        
        # reward_dribbling_ball_vel: encourage ball vel align with unit vector between ball target(goal) and ball current position
        ball_vel_2d = ball_vel[..., :2]
        d_ball_vel_2d = ball_vel_2d / torch.norm(ball_vel_2d, dim=-1, keepdim=True)
        d_ball_vel_2d = torch.nan_to_num(d_ball_vel_2d, nan=0.0)
        ball_goal_error = torch.norm(d_ball_vel_2d, dim=-1) - torch.sum(d_ball_vel_2d * d_ball_goal, dim=-1)
        
        delta_ball_goal = 2.0
        _rew_ball_goal = torch.exp(-delta_ball_goal * ball_goal_error)
        
        # reward_dribbling_robot_ball_yaw: 鼓励机器狗的朝向与球的朝向一致
        roll, pitch, yaw = get_euler_xyz(base_quat)
        body_yaw_vec = torch.zeros(self.num_envs, 2, device=self.device)
        body_yaw_vec[:, 0] = torch.cos(yaw)
        body_yaw_vec[:, 1] = torch.sin(yaw)
        robot_ball_body_yaw_error = torch.norm(body_yaw_vec, dim=-1) - torch.sum(d_robot_ball * body_yaw_vec, dim=-1)

        delta_dribbling_robot_ball_cmd_yaw = 2.0
        _rew_robot_ball_yaw = torch.exp(-delta_dribbling_robot_ball_cmd_yaw * robot_ball_body_yaw_error)
        
        reward = _rew_goal * 100 +\
                 _rew_robot_ball_vel * 5 +\
                 _rew_robot_ball_pos * 5 +\
                 _rew_ball_goal * 10 +\
                 _rew_robot_ball_goal * 3 +\
                 _rew_ball_to_goal * 10 +\
                 _rew_robot_ball_yaw * 3 + \
                 -self.reward_buffer["step count"] * 0.01
                
        return obs, reward, termination, info