import inspect

class BaseConfig:
    def __init__(self) -> None:
        """ Initializes all member classes recursively. Ignores all namse starting with '__' (buit-in methods)."""
        self.init_member_classes(self)
    
    @staticmethod
    def init_member_classes(obj):
        # iterate over all attributes names
        for key in dir(obj):
            # disregard builtin attributes
            # if key.startswith("__"):
            if key=="__class__":
                continue
            # get the corresponding attribute object
            var =  getattr(obj, key)
            # check if it the attribute is a class
            if inspect.isclass(var):
                # instantate the class
                i_var = var()
                # set the attribute to the instance instead of the type
                setattr(obj, key, i_var)
                # recursively init members of the attribute
                BaseConfig.init_member_classes(i_var)


class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    
    class policy:
        class_name = 'ActorCritic'  # ActorCritic, ActorCriticRecurrent
        # for MLP i.e. `ActorCritic`
        init_noise_std = 1.0
        # actor_hidden_dims = [512, 256, 128]
        # critic_hidden_dims = [512, 256, 128]
        actor_hidden_dims = [128, 64]
        critic_hidden_dims = [128, 64]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64   # default: 512
        rnn_num_layers = 1
        
    class algorithm:
        class_name = 'PPO'
        # training params
        # -- value function
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        # -- surrogate loss
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        entropy_coef = 0.01
        # -- training
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs * num_steps / num_mini_batches
        

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        
        num_steps_per_env = 24 # number of steps per environment per iteration
        max_iterations = 1500 # number of policy updates
        empirical_normalization = False
        # -- logging parameters
        save_interval = 50 # check for potential saves every `save_interval` iterations
        experiment_name = 'test'
        run_name = ''
        # -- logging writer
        logger = 'tensorboard'  # tensorboard, neptune, wandb
        neptune_project =  'legged_gym'
        wandb_project = 'legged_gym'
        wandb_entity = 'xander2077'
        # -- load and resuming
        resume = False
        load_run = -1 # -1 means load latest run
        checkpoint = -1 # -1 means load latest checkpoint
        resume_path = None # updated from load_run and chkpt
        
        
