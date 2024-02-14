import gym

class EmptyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.num_envs = self.env.num_envs
        self.num_agents = self.env.num_agents

        if hasattr(env.cfg.terrain, "BarrierTrack_kwargs"):
            self.BarrierTrack_kwargs = env.cfg.terrain.BarrierTrack_kwargs

        for _, key in enumerate(dir(self.env.cfg.rewards.scales)):
            if key[0] != "_" and "scale" in key:
                self.__setattr__(key, getattr(self.env.cfg.rewards.scales, key))