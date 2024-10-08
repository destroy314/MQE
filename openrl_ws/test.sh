python ./openrl_ws/test.py \
    --task go1football-defender \
    --algo ppo \
    --sim_device cuda:0 \
    --rl_device cuda:0 \
    --num_envs 10 \
    --checkpoint /home/zdj/Codes/multiagent-quadruped-environment/checkpoints/go1football-defender/ppo/ppo_90004500_steps/module.pt \
    # --record_video