### Go1Football 1v1

env:
    mqe_openrl_wrapper（统一接口，将obs转到cpu上，将action转到cuda上）
    [SingleAgentWrapper]（多智能体时如果需要只训练其中的一个agent）
    Go1FootballGameWrapper（根据任务不同不一样）

    mqe.envs.npc.go1_object.Go1Object（根据NPC不同不一样）

    mqe.envs.go1.go1.Go1
    mqe.envs.field.legged_robot_field.LeggedRobotField
    mqe.envs.base.legged_robot.LeggedRobot
    mqe.envs.base.base_task.BaseTask

    


"class": Go1Object



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
