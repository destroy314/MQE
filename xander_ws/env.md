单个环境由mqe.envs.utils中的make_mqe_env定义，每个环境由三个类组成：
- mqe.envs.npc
- mqe.envs.configs
- mqe.envs.wrappers



env:
    mqe_openrl_wrapper（统一接口，将obs转到cpu上，将action转到cuda上）
    [SingleAgentWrapper]（多智能体时如果需要只训练其中的一个agent）
    Go1FootballGameWrapper（根据任务不同不一样）

    mqe.envs.npc.go1_object.Go1Object（根据NPC不同不一样）

    mqe.envs.go1.go1.Go1
    mqe.envs.field.legged_robot_field.LeggedRobotField
    mqe.envs.base.legged_robot.LeggedRobot
    mqe.envs.base.base_task.BaseTask

    


"class": 
- Go1Object(Go1)
  定义了npc(ball...)的prepare/create函数，无调用

- Go1(LeggedRobotField)  主要的定义都在这里，包括Go1的locomotion policy
  


- LeggedRobotField(LeggedRobot)





- LeggedRobot(BaseTask): 主体函数
  step()
    - pre_physics_step()
    - render()
    - _compute_torques()
    - set_dof_actuation_force_tensor() 施加扭矩，分为self.decimation个
    - post_physics_step()
      - check_termination
      - compute_reward
      - _step_npc
      - reset_idx
      - compute_observations

  _create_envs()


- BaseTask  主要是仿真gym的定义
  obs_buf: (num_envs, num_obs)
  rew_buf: (num_envs*num_agents)
  reset_buf: (num_envs,)
  episode_length_buf: (num_envs,)
  time_out_buf: (num_envs,)
  collide_buf: (num_envs,)

  reset(): 调用一次reset_idx, 再调用一次step, return obs, privileged_obs

"config": 
1. Go1Football1vs1Cfg


2. Go1Cfg
    1. class obs
        self.root_states: (num_envs * num_agents, 13)




"wrapper": 
1. Go1FootballGameWrapper
    num_agents: 2
    Obs: shape (20,)
    Action: shape (3,)

    reset, step 没有obs返回


2. mqe_openrl_wrapper
    action: Box(-1.0, 1.0, (3,), float64)
    obs: Box(-inf, inf, (20,), float64)

    reset: obs.cpu().numpy()
    step:  obs, rewards, dones, infos
