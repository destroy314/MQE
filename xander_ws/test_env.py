from mqe.envs.utils import custom_cfg, make_mqe_env
from openrl_ws.utils import get_args
import torch

if __name__ == '__main__':
    import debugpy; debugpy.connect(50678)
    # debugpy.listen(12361)
    # print('wait debugger')
    # debugpy.wait_for_client()
    # print("Debugger Attached")

    args = get_args()

    # task_name = "go1plane"
    # task_name = "go1gate"
    # task_name = "go1football-defender"
    # task_name = "go1football-1vs1"
    # task_name = "go1football-2vs2"
    # task_name = "go1sheep-easy"
    # task_name = "go1sheep-hard"
    # task_name = "go1seesaw"
    # task_name = "go1door"
    # task_name = "go1pushbox"
    # task_name = "go1tug"
    # task_name = "go1wrestling"
    # task_name = "go1rotationdoor"
    # task_name = "go1bridge"
    task_name = "go1football-shoot"

    args.num_envs = 1
    args.headless = False

    env, env_cfg = make_mqe_env(task_name, args, custom_cfg(args))
    env.reset()
    # breakpoint()
    action_sample = torch.tensor(env.action_space.sample())
    
    step = 0

    while True:
        obs, reward, done, info = env.step(torch.randn_like(action_sample, dtype=torch.float32, device="cuda").repeat(env.num_envs, env.num_agents, 1))

        step += 1
        print(f"step: {step}")
