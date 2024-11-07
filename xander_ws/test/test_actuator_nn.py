import torch

actuator_network = torch.jit.load("/home/zdj/Codes/multiagent-quadruped-environment/resources/actuator_nets/unitree_go1.pt", map_location="cuda:0")

print(actuator_network)