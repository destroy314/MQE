from rsl_rl.rsl_rl.env import VecEnv
from rsl_rl.rsl_rl.runners import OnPolicyRunner

from mqe.utils import get_args
from mqe.envs.utils import make_mqe_env, custom_cfg




if __name__ == '__main__':
    env_args = get_args()
    args = 
    
    
    
    
    
    
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
    #save resume path before creating a new log_dir
    resume = train_cfg.runner.resume
    if resume:
        # load previously trained model
        resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
        print(f"Loading model from: {resume_path}")
        runner.load(resume_path)
    
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)