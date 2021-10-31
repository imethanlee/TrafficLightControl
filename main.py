from sumo.sumo_agent import *
from sumo.fixed_policy_simulate import *
import argparse
from net_agent import NetAgent


# TODO: 参考下面这个method来 1.获取state 2.执行action 3.获取reward
def dqn_simulate(args):
    # Specify the traffic environment
    sumo_agent = SumoAgent(args.test_case_name,
                           require_gui=args.require_gui,
                           gamma=args.gamma,
                           delta_t=args.delta_t)

    net_agent = NetAgent(args)
    run_counts = args.run_counts
    for epoch in range(args.max_epoch):

        # Start interacting with the SUMO environment
        sumo_agent.sumo_start()
        step = 0
        net_agent.reset_update_count()

        current_time = sumo_agent.get_current_time()
        while current_time < run_counts:
            """#####################################################"""
            """####   1. Get the current state                  ####"""
            """#####################################################"""

            current_state = sumo_agent.get_current_state_without_update()

            action = net_agent.choose(current_time, current_state)

            """#####################################################"""
            """####   2. Take your action                       ####"""
            """#####################################################"""
            """ (a boolean torch.Tensor/list with length num_junctions) """

            # get reward from sumo agent
            sumo_agent.step(action)

            """#####################################################"""
            """####  3. Get step/cumulative reward/next state   ####"""
            """#####################################################"""

            next_state = sumo_agent.get_current_state_with_update()
            current_reward = sumo_agent.get_current_reward()
            # cumulative_reward = sumo_agent.get_cumulative_reward()

            net_agent.remember(current_state, action, current_reward, next_state)

            net_agent.trainer()
            """#####################################################"""
            """########                END              ###########"""
            """#####################################################"""
            if sumo_agent.all_travels_completed():
                break
            step += sumo_agent.delta_t
        sumo_agent.sumo_end()
    # End interacting with the SUMO environment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Test Case Name List
    # 1. "./data/self_defined_1/osm": small-scale self-defined network
    # 2. "./data/nus1/osm": NUS network 1
    # 3. "./data/nus2/osm": NUS network 2
    parser.add_argument('--require_gui', type=bool, default=True)
    parser.add_argument('--test_case_name', type=str, default="./data/self_defined_1/osm")
    parser.add_argument('--num_phases', type=int, default=2)
    parser.add_argument('--num_actions', type=int, default=16)
    parser.add_argument('--state_dim', type=int, default=20)
    parser.add_argument('--memory_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    # parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max_epoch', default=50, type=int)

    parser.add_argument('--run_counts', default=72000, type=int)

    parser.add_argument('--delta_t', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.8)

    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # dqn_simulate()
    # fixed_policy_simulate(args)
    torch.manual_seed(args.seed)
    dqn_simulate(args)
    # np.random.seed(opt.seed)