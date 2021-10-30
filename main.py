from sumo.sumo_agent import *
from sumo.fixed_policy_simulate import *
import argparse

parser = argparse.ArgumentParser()
# Test Case Name List
# 1. "./data/self_defined_1/osm": small-scale self-defined network
# 2. "./data/nus1/osm": NUS network 1
# 3. "./data/nus2/osm": NUS network 2
parser.add_argument('--require_gui', type=bool, default=True)
parser.add_argument('--test_case_name', type=str, default="./data/self_defined_1/osm")
parser.add_argument('--delta_t', type=int, default=5)
parser.add_argument('--gamma', type=float, default=0.8)
args = parser.parse_args()


# TODO: 参考下面这个method来 1.获取state 2.执行action 3.获取reward
def dqn_simulate():
    # Specify the traffic environment
    sumo_agent = SumoAgent(args.test_case_name,
                           require_gui=args.require_gui,
                           gamma=args.gamma,
                           delta_t=args.delta_t)

    # Start interacting with the SUMO environment
    sumo_agent.sumo_start()
    step = 0
    while True:
        """#####################################################"""
        """####   1. Get the current state                  ####"""
        """#####################################################"""

        current_state = sumo_agent.get_current_state_without_update()

        """#####################################################"""
        """####   2. Take your action                       ####"""
        """#####################################################"""
        """ (a boolean torch.Tensor/list with length num_junctions) """

        # action = torch.Tensor([0 for _ in range(sumo_agent.get_num_junctions())])
        action = None
        sumo_agent.step(action)

        """#####################################################"""
        """####  3. Get step/cumulative reward/next state   ####"""
        """#####################################################"""

        next_state = sumo_agent.get_current_state_with_update()
        # current_reward = sumo_agent.get_current_reward()
        # cumulative_reward = sumo_agent.get_cumulative_reward()

        """#####################################################"""
        """########                END              ###########"""
        """#####################################################"""
        if sumo_agent.all_travels_completed():
            break
        step += sumo_agent.delta_t
    sumo_agent.sumo_end()
    # End interacting with the SUMO environment


# dqn_simulate()
fixed_policy_simulate(args)
